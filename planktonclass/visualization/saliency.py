# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities to compute SaliencyMasks."""
import numpy as np
import tensorflow as tf


class SaliencyMask(object):
    """Base class for saliency masks. Alone, this class doesn't do anything."""

    def __init__(self, model, output_index=0):
        """Constructs a SaliencyMask.

        Args:
            model: the keras model used to make prediction
            output_index: the index of the node in the last layer to take derivative on
        """

    def get_mask(self, input_image):
        """Returns an unsmoothed mask.

        Args:
            input_image: input image with shape (H, W, 3).
        """

    def get_smoothed_mask(self, input_image, stdev_spread=0.2, nsamples=50):
        """Returns a mask that is smoothed with the SmoothGrad method.

        Args:
            input_image: input image with shape (H, W, 3).
        """
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples


class GradientSaliency(SaliencyMask):
    r"""A SaliencyMask class that computes saliency masks with a gradient."""

    def __init__(self, model, output_index=0):
        self.model = model
        self.output_index = output_index

    def get_mask(self, input_image):
        """Returns a vanilla gradient mask.

        Args:
            input_image: input image with shape (H, W, 3).
        """
        # Add batch dimension
        x_value = np.expand_dims(input_image, axis=0).astype(np.float32)
        x_tensor = tf.convert_to_tensor(x_value)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            outputs = self.model(x_tensor, training=False)
            # Handle multiple outputs
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs
            # Select the required output index (assumes batch dimension first)
            loss = output[:, self.output_index]
        gradients = tape.gradient(loss, x_tensor)
        # Remove batch dimension and convert to numpy
        gradients_np = gradients[0].numpy()
        return gradients_np
