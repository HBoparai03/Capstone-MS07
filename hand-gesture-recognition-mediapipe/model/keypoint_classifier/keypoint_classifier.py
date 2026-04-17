#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        # Performance: Use multi-threading for faster inference
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self._input_index = self.input_details[0]['index']
        self._output_index = self.output_details[0]['index']
        self._input_buffer = np.zeros(self.input_details[0]['shape'], dtype=np.float32)

    def __call__(
        self,
        landmark_list,
    ):
        np.copyto(self._input_buffer[0], np.asarray(landmark_list, dtype=np.float32), casting='unsafe')
        self.interpreter.set_tensor(self._input_index, self._input_buffer)
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(self._output_index)[0]
        return int(np.argmax(result))
