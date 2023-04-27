#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""


import argparse
import logging
from typing import Tuple

import numpy as np
import torch

import time
import os

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GenerationMixin,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    GPT2Config,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


#
# Functions to prepare models' input
#


def prepare_ctrl_input(args, _, tokenizer, prompt_text):
    if args.temperature > 0.7:
        logger.info("CTRL typically works better with lower temperatures (and lower top_k).")

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not any(encoded_prompt[0] == x for x in tokenizer.control_codes.values()):
        logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
    return prompt_text


def prepare_xlm_input(args, model, tokenizer, prompt_text):
    # kwargs = {"language": None, "mask_token_id": None}

    # Set the language
    use_lang_emb = hasattr(model.config, "use_lang_emb") and model.config.use_lang_emb
    if hasattr(model.config, "lang2id") and use_lang_emb:
        available_languages = model.config.lang2id.keys()
        if args.xlm_language in available_languages:
            language = args.xlm_language
        else:
            language = None
            while language not in available_languages:
                language = input("Using XLM. Select language in " + str(list(available_languages)) + " >>> ")

        model.config.lang_id = model.config.lang2id[language]
        # kwargs["language"] = tokenizer.lang2id[language]

    # TODO fix mask_token_id setup when configurations will be synchronized between models and tokenizers
    # XLM masked-language modeling (MLM) models need masked token
    # is_xlm_mlm = "mlm" in args.model_name_or_path
    # if is_xlm_mlm:
    #     kwargs["mask_token_id"] = tokenizer.mask_token_id

    return prompt_text


def prepare_xlnet_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


def prepare_transfoxl_input(args, _, tokenizer, prompt_text):
    prefix = args.prefix if args.prefix else args.padding_text if args.padding_text else PREFIX
    prompt_text = prefix + prompt_text
    return prompt_text


PREPROCESSING_FUNCTIONS = {
    "ctrl": prepare_ctrl_input,
    "xlm": prepare_xlm_input,
    "xlnet": prepare_xlnet_input,
    "transfo-xl": prepare_transfoxl_input,
}


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)


def sparse_model_config(model_config):
    embedding_size = None
    if hasattr(model_config, "hidden_size"):
        embedding_size = model_config.hidden_size
    elif hasattr(model_config, "n_embed"):
        embedding_size = model_config.n_embed
    elif hasattr(model_config, "n_embd"):
        embedding_size = model_config.n_embd

    num_head = None
    if hasattr(model_config, "num_attention_heads"):
        num_head = model_config.num_attention_heads
    elif hasattr(model_config, "n_head"):
        num_head = model_config.n_head

    if embedding_size is None or num_head is None or num_head == 0:
        raise ValueError("Check the model config")

    num_embedding_size_per_head = int(embedding_size / num_head)
    num_layer = model_config.n_layer

    return num_layer, num_head, num_embedding_size_per_head


def prepare_jit_inputs(inputs, model, tokenizer):
    num_batch = len(inputs)
    dummy_input = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    num_block_layers, num_attention_heads, num_embedding_size_per_head = sparse_model_config(model.config)
    if model.config.model_type == "bloom":
        past_key_values = tuple(
            (
                torch.zeros(int(num_attention_heads * num_batch), num_embedding_size_per_head, 1)
                .to(model.config.torch_dtype)
                .to(model.device),
                torch.zeros(int(num_attention_heads * num_batch), 1, num_embedding_size_per_head)
                .to(model.config.torch_dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )
    else:
        past_key_values = tuple(
            (
                torch.zeros(num_batch, num_attention_heads, 1, num_embedding_size_per_head)
                .to(model.config.torch_dtype)
                .to(model.device),
                torch.zeros(num_batch, num_attention_heads, 1, num_embedding_size_per_head)
                .to(model.config.torch_dtype)
                .to(model.device),
            )
            for _ in range(num_block_layers)
        )

    dummy_input["attention_mask"] = torch.cat(
        [
            torch.zeros(dummy_input["attention_mask"].shape[0], 1).to(dummy_input["attention_mask"].dtype),
            dummy_input["attention_mask"],
        ],
        -1,
    )

    if model.config.use_cache:
        jit_inputs = (
            dummy_input["input_ids"].to(model.device),
            past_key_values,
            dummy_input["attention_mask"].to(model.device),
        )
    else:
        jit_inputs = (
            dummy_input["input_ids"].to(model.device),
            dummy_input["attention_mask"].to(model.device),
        )

    return jit_inputs


class _ModelFallbackWrapper(GenerationMixin):
    __slots__ = ("_optimized", "_default")

    def __init__(self, optimized, default):
        self._optimized = optimized
        self._default = default

    def __call__(self, *args, **kwargs):
        if kwargs["past_key_values"] is None:
            return self._default(*args, **kwargs)
        trace_graph_inputs = []
        kwargs.pop("position_ids", None)
        for k, v in kwargs.items():
            if v is not None and not isinstance(v, bool):
                trace_graph_inputs.append(v)
        trace_graph_inputs = tuple(trace_graph_inputs)
        outputs = self._optimized(*trace_graph_inputs)
        lm_logits = outputs[0]
        past_key_values = outputs[1]
        fixed_output = CausalLMOutputWithPast(
            loss=None,
            logits=lm_logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )
        return fixed_output

    def __getattr__(self, item):
        return getattr(self._default, item)

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs
    ):
        return self._default.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PretrainedModel.beam_search`] or
        [`~PretrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return self._default._reorder_cache(past_key_values, beam_idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--prompt", type=str, default="hello world")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")
    parser.add_argument("--padding_text", type=str, default="", help="Deprecated, the use of `--prefix` is preferred.")
    parser.add_argument("--xlm_language", type=str, default="", help="Optional language when used with the XLM model.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--num_warmup", "--warmup_iter", type=int, default=50, help="The number warmup, default is 50.")
    parser.add_argument("--num_iters", "--early_stop_at_iter",type=int, default=500, help="The number iters of benchmark, default is 500.")
    parser.add_argument("--jit", action="store_true", help="Use jit optimize to do optimization.")
    parser.add_argument("--nv_fuser", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda", "xpu"], default="cpu", type=str)
    parser.add_argument("--channels_last", type=bool, default=False, help="Use pytorch NHWC.")
    parser.add_argument("--profile", action="store_true", default=False, help="Trigger profile on current topology.")
    parser.add_argument('--precision', default='float32', help='Precision, "float32" or "bfloat16"')
    parser.add_argument('--do_eval', action="store_true", help='do evaluation')
    parser.add_argument("--overwrite_output_dir", action="store_true", )
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--per_device_eval_batch_size", type=int, default="")
    args = parser.parse_args()

    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    if args.device == "xpu":
        import intel_extension_for_pytorch
    elif args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    args.num_return_sequences = args.per_device_eval_batch_size
    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    #config = GPT2Config.from_pretrained(args.model_name_or_path)
    #config.precision = args.precision
    #model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")

    # Different models need different input formatting and/or extra arguments
    requires_preprocessing = args.model_type in PREPROCESSING_FUNCTIONS.keys()
    if requires_preprocessing:
        prepare_input = PREPROCESSING_FUNCTIONS.get(args.model_type)
        preprocessed_prompt_text = prepare_input(args, model, tokenizer, prompt_text)

        if model.__class__.__name__ in ["TransfoXLLMHeadModel"]:
            tokenizer_kwargs = {"add_space_before_punct_symbol": True}
        else:
            tokenizer_kwargs = {}

        encoded_prompt = tokenizer.encode(
            preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt", **tokenizer_kwargs
        )
    else:
        prefix = args.prefix if args.prefix else args.padding_text
        encoded_prompt = tokenizer.encode(prefix + prompt_text, add_special_tokens=False, return_tensors="pt")

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    def forward_loop(args, encoded_prompt, fuser_mode):
        # inference benchmark
        total_time = 0.0
        total_sample = 0
        batch_time_list = []
        profile_iter = args.num_iters // 2
        if args.profile and args.device == "xpu":
            for i in range(args.num_iters):
                tic  = time.time()
                with torch.autograd.profiler_legacy.profile(enabled=args.profile, use_xpu=args.xpu, record_shapes=False) as prof:
                    encoded_prompt = encoded_prompt.to(args.device)
                    output_sequences = model.generate(
                        input_ids=encoded_prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=args.p,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                    torch.xpu.synchronize()
                toc = time.time()
                print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                if i >= args.num_warmup:
                    total_time += (toc - tic)
                    total_sample += args.num_return_sequences
                    batch_time_list.append((toc - tic) * 1000)
                if args.profile and i == profile_iter:
                    import pathlib
                    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                    if not os.path.exists(timeline_dir):
                        try:
                            os.makedirs(timeline_dir)
                        except:
                            pass
                    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"),
                        timeline_dir+'profile.pt')
                    torch.save(prof.key_averages(group_by_input_shape=True).table(),
                        timeline_dir+'profile_detail.pt')
        elif args.profile and args.device == "cuda":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_iter,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i in range(args.num_iters):
                    tic  = time.time()
                    encoded_prompt = encoded_prompt.to(args.device)
                    with torch.jit.fuser(fuser_mode):
                        output_sequences = model.generate(
                            input_ids=encoded_prompt,
                            max_length=args.length + len(encoded_prompt[0]),
                            temperature=args.temperature,
                            top_k=args.k,
                            top_p=args.p,
                            repetition_penalty=args.repetition_penalty,
                            do_sample=True,
                            num_return_sequences=args.num_return_sequences,
                        )
                    torch.cuda.synchronize()
                    toc = time.time()
                    p.step()
                    print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                    if i >= args.num_warmup:
                        total_time += (toc - tic)
                        total_sample += args.num_return_sequences
                        batch_time_list.append((toc - tic) * 1000)
        elif args.profile and args.device == "cpu":
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                schedule=torch.profiler.schedule(
                    wait=profile_iter,
                    warmup=2,
                    active=1,
                ),
                on_trace_ready=trace_handler,
            ) as p:
                for i in range(args.num_iters):
                    tic  = time.time()
                    encoded_prompt = encoded_prompt.to(args.device)
                    output_sequences = model.generate(
                        input_ids=encoded_prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=args.p,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                    toc = time.time()
                    p.step()
                    print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                    if i >= args.num_warmup:
                        total_time += (toc - tic)
                        total_sample += args.num_return_sequences
                        batch_time_list.append((toc - tic) * 1000)
        elif not args.profile and args.device == "cuda":
            for i in range(args.num_iters):
                tic  = time.time()
                encoded_prompt = encoded_prompt.to(args.device)
                with torch.jit.fuser(fuser_mode):
                    output_sequences = model.generate(
                        input_ids=encoded_prompt,
                        max_length=args.length + len(encoded_prompt[0]),
                        temperature=args.temperature,
                        top_k=args.k,
                        top_p=args.p,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=True,
                        num_return_sequences=args.num_return_sequences,
                    )
                torch.cuda.synchronize()
                toc = time.time()
                print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                if i >= args.num_warmup:
                    total_time += (toc - tic)
                    total_sample += args.num_return_sequences
                    batch_time_list.append((toc - tic) * 1000)
        else:
            for i in range(args.num_iters):
                tic  = time.time()
                encoded_prompt = encoded_prompt.to(args.device)
                output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.length + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                )
                if args.device == "xpu":
                    torch.xpu.synchronize()
                toc = time.time()
                print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                if i >= args.num_warmup:
                    total_time += (toc - tic)
                    total_sample += args.num_return_sequences
                    batch_time_list.append((toc - tic) * 1000)

        print("\n", "-"*20, "Summary", "-"*20)
        print("batch size: ", args.num_return_sequences)
        latency = total_time / total_sample * 1000
        throughput = total_sample / total_time
        print("Latency:\t {:.3f} ms".format(latency))
        print("Throughput:\t {:.2f} samples/s".format(throughput))
        # P50
        batch_time_list.sort()
        p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
        p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
        p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
        print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
                % (p50_latency, p90_latency, p99_latency))

    model.eval()
    if args.nv_fuser:
        print("---- Use trace model.")
        fuser_mode = "fuser2"
    else:
        fuser_mode = "none"
    with torch.no_grad():
        if args.device == "xpu":
            datatype = torch.float16 if args.precision == "float16" else torch.bfloat16 if args.precision == "bfloat16" else torch.float
            model = torch.xpu.optimize(model=model, dtype=datatype)
        if args.precision == "float16" and args.device == "cuda":
            print("---- Use autocast fp16 cuda")
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                forward_loop(args, encoded_prompt, fuser_mode)
        elif args.precision == "float16" and args.device == "xpu":
            print("---- Use autocast fp16 xpu")
            with torch.xpu.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                forward_loop(args, encoded_prompt, fuser_mode)
        elif args.precision == "bfloat16" and args.device == "cpu":
            print("---- Use autocast bf16 cpu")
            with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
                forward_loop(args, encoded_prompt, fuser_mode)
        elif args.precision == "bfloat16" and args.device == "xpu":
            print("---- Use autocast bf16 xpu")
            with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                forward_loop(args, encoded_prompt, fuser_mode)
        else:
            print("---- no autocast")
            forward_loop(args, encoded_prompt, fuser_mode)
    
    return
#=======
#    if args.jit:
#        jit_input_texts = ["jit"]
#        jit_inputs = prepare_jit_inputs(jit_input_texts, model, tokenizer)
#        torch._C._jit_set_texpr_fuser_enabled(False)
#        model.config.return_dict = False
#        traced_model = torch.jit.trace(model, jit_inputs, strict=False)
#        traced_model = torch.jit.freeze(traced_model.eval())
#        traced_model(*jit_inputs)
#        traced_model(*jit_inputs)
#
#        model = _ModelFallbackWrapper(traced_model, model)
#
#    output_sequences = model.generate(
#        input_ids=input_ids,
#        max_length=args.length + len(encoded_prompt[0]),
#        temperature=args.temperature,
#        top_k=args.k,
#        top_p=args.p,
#        repetition_penalty=args.repetition_penalty,
#        do_sample=True,
#        num_return_sequences=args.num_return_sequences,
#    )
#>>>>>>> upstream/v4.28-release

    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        #print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token
        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
        )

        generated_sequences.append(total_sequence)
        #print(total_sequence)

    return generated_sequences


if __name__ == "__main__":
    main()
