#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class PointHistoryClassifier(object):
    def __init__(
        self,
        model_path='model/point_history_classifier/point_history_classifier.tflite',
        score_th=0.5,
        invalid_value=0,
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

        self.score_th = score_th
        self.invalid_value = invalid_value

    def __call__(
        self,
        point_history,
    ):
        np.copyto(self._input_buffer[0], np.asarray(point_history, dtype=np.float32), casting='unsafe')
        self.interpreter.set_tensor(self._input_index, self._input_buffer)
        self.interpreter.invoke()

        result = self.interpreter.get_tensor(self._output_index)[0]
        result_index = int(np.argmax(result))

        if result[result_index] < self.score_th:
            result_index = self.invalid_value

        return result_index
