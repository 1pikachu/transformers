from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import time


class ExampleHook(Callback):
    def __init__(self, tensorboard=False):
        self.train_total_batch = 0
        self.train_batch = 0
        self.train_total_time = 0
        self.test_total_batch = 0
        self.test_batch = 0
        self.test_total_time = 0
        self.predict_total_batch = 0
        self.predict_batch = 0
        self.predict_total_time = 0
        self.tensorboard = tensorboard

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_train_batch_begin(self, batch, logs={}):
        self.time = time.time()
        if self.tensorboard and self.train_total_batch == 3:
            print("---- collect tensorboard")
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
            tf.profiler.experimental.start('./tensorboard_data', options = options)
        return

    def on_train_batch_end(self, batch, logs={}):
        duration = time.time() - self.time
        if batch > 10:
            self.train_total_time += duration
            self.train_batch += 1
        if self.tensorboard and self.train_total_batch == 3:
            tf.profiler.experimental.stop()
            print("---- collect tensorboard end")
        self.train_total_batch += 1
        print("Iteration: {}, training time: {}".format(self.train_total_batch, duration), flush=True)
        return

    def on_test_batch_begin(self, batch, logs={}):
        self.time = time.time()
        if self.tensorboard and self.test_total_batch == 3:
            print("---- collect tensorboard")
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
            tf.profiler.experimental.start('./tensorboard_data', options = options)
        return

    def on_test_batch_end(self, batch, logs={}):
        duration = time.time() - self.time
        if batch > 10:
            self.test_total_time += duration
            self.test_batch += 1
        if self.tensorboard and self.test_total_batch == 3:
            tf.profiler.experimental.stop()
            print("---- collect tensorboard end")
        self.test_total_batch += 1
        print("Iteration: {}, inference time: {}".format(self.test_total_batch, duration), flush=True)
        return

    def on_predict_batch_begin(self, batch, logs={}):
        self.time = time.time()
        if self.tensorboard and self.predict_total_batch == 3:
            print("---- collect tensorboard")
            options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)
            tf.profiler.experimental.start('./tensorboard_data', options = options)
        return

    def on_predict_batch_end(self, batch, logs={}):
        duration = time.time() - self.time
        if batch > 10:
            self.predict_total_time += duration
            self.predict_batch += 1
        if self.tensorboard and self.predict_total_batch == 3:
            tf.profiler.experimental.stop()
            print("---- collect tensorboard end")
        self.predict_total_batch += 1
        print("Iteration: {}, inference time: {}".format(self.predict_total_batch, duration), flush=True)
        return

class KerasHook(Callback):
    def __init__(self):
        self.test_total_batch = 0
        self.test_batch = 0
        self.test_total_time = 0
        self.predict_total_batch = 0
        self.predict_batch = 0
        self.predict_total_time = 0

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        return

    def on_test_batch_begin(self, batch, logs={}):
        self.time = time.time()
        return

    def on_test_batch_end(self, batch, logs={}):
        duration = time.time() - self.time
        if batch > 10:
            self.test_total_time += duration
            self.test_batch += 1
        self.test_total_batch += 1
        print("Iteration: {}, inference time: {}".format(self.test_total_batch, duration), flush=True)
        return

    def on_predict_batch_begin(self, batch, logs={}):
        self.time = time.time()
        return

    def on_predict_batch_end(self, batch, logs={}):
        duration = time.time() - self.time
        if batch > 10:
            self.predict_total_time += duration
            self.predict_batch += 1
        self.predict_total_batch += 1
        print("Iteration: {}, inference time: {}".format(self.predict_total_batch, duration), flush=True)
        return
