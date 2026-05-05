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
"""Utilites to computed GuidedBackprop SaliencyMasks"""

import numpy as np

import tensorflow as tf

from .saliency import SaliencyMask


class GuidedBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with GuidedBackProp.

    This implementation copies the TensorFlow graph to a new graph with the ReLU
    gradient overwritten as in the paper:
    https://arxiv.org/abs/1412.6806
    """

    GuidedReluRegistered = False

    def __init__(self, model, output_index=0, custom_loss=None):
        """
        Constructs a GuidedBackprop SaliencyMask.

        Args:
            model: A tf.keras.Model instance.
            output_index: Index of the output neuron to compute gradients for.
            custom_loss: (Optional) Custom loss function, if needed for loading.
        """
        self.output_index = output_index

        # Define the custom GuidedReLU activation using @tf.custom_gradient
        @tf.custom_gradient
        def guided_relu(x):

            def grad(dy):
                gate_f = tf.cast(x > 0, dtype=dy.dtype)
                gate_r = tf.cast(dy > 0, dtype=dy.dtype)
                return dy * gate_f * gate_r

            return tf.nn.relu(x), grad

        # Function to replace relu activations with guided_relu
        def replace_relu(layer):
            if (hasattr(layer, "activation")
                    and layer.activation == tf.keras.activations.relu):
                layer.activation = guided_relu
            return layer

        # Clone the model, swapping out relu for guided_relu
        self.guided_model = tf.keras.models.clone_model(
            model, clone_function=replace_relu)
        self.guided_model.set_weights(model.get_weights())

    def get_mask(self, input_image):
        """
        Computes the guided backpropagation saliency mask for a given input image.

        Args:
            input_image: Input image as a NumPy array (H, W, C) or (H, W, ...).

        Returns:
            A NumPy array of the same shape as input_image with the guided gradients.
        """
        # Add batch dimension and ensure float32 dtype
        x_value = np.expand_dims(input_image, axis=0).astype(np.float32)
        x_tensor = tf.convert_to_tensor(x_value)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            outputs = self.guided_model(x_tensor, training=False)
            # Handle multi-output models
            if isinstance(outputs, (list, tuple)):
                output = outputs[0]
            else:
                output = outputs
            # Compute loss for the selected output index
            loss = output[:, self.output_index]
        gradients = tape.gradient(loss, x_tensor)
        return gradients[0].numpy()
