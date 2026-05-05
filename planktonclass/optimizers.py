"""
Custom optimizers to implement lr_mult as in caffe

Date: April 2025
Last updated: March 2026
Original Author: Ignacio Heredia (CSIC)
Updated for TensorFlow 2.10 and maintained by: Wout Decrop (VLIZ)
Contact: wout.decrop@vliz.be
Github: ai4os-hub / phyto-plankton-classification

References
----------
https://github.com/keras-team/keras/issues/5920#issuecomment-328890905
"""

import tensorflow as tf
from tensorflow.keras import optimizers


class customSGD(optimizers.SGD):
    """
    Custom subclass of the SGD optimizer to implement lr_mult as in Caffe
    """

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        lr_mult=0.1,
        excluded_vars=[],
        name="customSGD",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            name=name,
            **kwargs,
        )
        self.lr_mult = lr_mult
        self.excluded_vars = excluded_vars

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Apply lr multiplier for vars outside excluded_vars
        if var.name in self.excluded_vars:
            multiplied_lr = lr_t
        else:
            multiplied_lr = lr_t * self.lr_mult

        momentum = self.momentum

        if apply_state is None:
            apply_state = {}
        momentum_var = self.get_slot(var, "momentum")

        if self.nesterov:
            momentum_buf = momentum * momentum_var - multiplied_lr * grad
            var_update = var.assign_add(momentum_buf * momentum -
                                        multiplied_lr * grad)
        else:
            momentum_buf = momentum * momentum_var - multiplied_lr * grad
            var_update = var.assign_add(momentum_buf)

        momentum_update = momentum_var.assign(momentum_buf)

        return tf.group(var_update, momentum_update)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # Implementation for sparse gradients
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Apply lr multiplier for vars outside excluded_vars
        if var.name in self.excluded_vars:
            multiplied_lr = lr_t
        else:
            multiplied_lr = lr_t * self.lr_mult

        momentum = self.momentum

        momentum_var = self.get_slot(var, "momentum")

        if self.nesterov:
            momentum_buf = (momentum * momentum_var.sparse_read(indices) -
                            multiplied_lr * grad)
            var_update = self._resource_scatter_add(
                var,
                indices,
                momentum_buf * momentum - multiplied_lr * grad,
            )
        else:
            momentum_buf = (momentum * momentum_var.sparse_read(indices) -
                            multiplied_lr * grad)
            var_update = self._resource_scatter_add(var, indices, momentum_buf)

        momentum_update = self._resource_scatter_update(
            momentum_var, indices, momentum_buf)

        return tf.group(var_update, momentum_update)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lr_mult": self.lr_mult,
            "excluded_vars": self.excluded_vars,
        })
        return config


class customAdam(optimizers.Adam):
    """
    Custom subclass of the Adam optimizer to implement lr_mult as in Caffe
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        lr_mult=0.1,
        excluded_vars=[],
        name="customAdam",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )
        self.lr_mult = lr_mult
        self.excluded_vars = excluded_vars

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Apply lr multiplier for vars outside excluded_vars
        if var.name in self.excluded_vars:
            multiplied_lr_t = lr_t
        else:
            multiplied_lr_t = lr_t * self.lr_mult

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = beta_1_t * m + (1.0 - beta_1_t) * grad
        v_t = beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)

        m_corr = m_t / (1.0 - tf.pow(beta_1_t, local_step))
        v_corr = v_t / (1.0 - tf.pow(beta_2_t, local_step))

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_corr)
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(vhat_t) + epsilon_t))
            vhat_update = vhat.assign(vhat_t)
            updates = [
                var_update,
                m.assign(m_t),
                v.assign(v_t),
                vhat_update,
            ]
        else:
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(v_corr) + epsilon_t))
            updates = [var_update, m.assign(m_t), v.assign(v_t)]

        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Apply lr multiplier for vars outside excluded_vars
        if var.name in self.excluded_vars:
            multiplied_lr_t = lr_t
        else:
            multiplied_lr_t = lr_t * self.lr_mult

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.cast(self.beta_1, var_dtype)
        beta_2_t = tf.cast(self.beta_2, var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = v.assign(v * beta_2_t)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_corr = m_t / (1.0 - tf.pow(beta_1_t, local_step))
        v_corr = v_t / (1.0 - tf.pow(beta_2_t, local_step))

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_corr)
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(vhat_t) + epsilon_t))
            vhat_update = vhat.assign(vhat_t)
            updates = [var_update, vhat_update]
        else:
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(v_corr) + epsilon_t))
            updates = [var_update]

        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update({
            "lr_mult": self.lr_mult,
            "excluded_vars": self.excluded_vars,
        })
        return config


class customAdamW(optimizers.Optimizer):
    """
    Custom AdamW optimizer with lr_mult as in Caffe
    """

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        weight_decay=0.025,
        amsgrad=False,
        lr_mult=0.1,
        excluded_vars=[],
        name="customAdamW",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay", weight_decay)
        self.epsilon = epsilon
        self.amsgrad = amsgrad
        self.lr_mult = lr_mult
        self.excluded_vars = excluded_vars

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "v")
            if self.amsgrad:
                self.add_slot(var, "vhat")

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Apply lr multiplier for vars outside excluded_vars
        if var.name in self.excluded_vars:
            multiplied_lr_t = lr_t
        else:
            multiplied_lr_t = lr_t * self.lr_mult

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.cast(self._get_hyper("beta_1"), var_dtype)
        beta_2_t = tf.cast(self._get_hyper("beta_2"), var_dtype)
        weight_decay = tf.cast(self._get_hyper("weight_decay"), var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = beta_1_t * m + (1.0 - beta_1_t) * grad
        v_t = beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad)

        m_corr = m_t / (1.0 - tf.pow(beta_1_t, local_step))
        v_corr = v_t / (1.0 - tf.pow(beta_2_t, local_step))

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_corr)
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(vhat_t) + epsilon_t) +
                                        multiplied_lr_t * weight_decay * var)
            vhat_update = vhat.assign(vhat_t)
            updates = [
                var_update,
                m.assign(m_t),
                v.assign(v_t),
                vhat_update,
            ]
        else:
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(v_corr) + epsilon_t) +
                                        multiplied_lr_t * weight_decay * var)
            updates = [var_update, m.assign(m_t), v.assign(v_t)]

        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)

        # Apply lr multiplier for vars outside excluded_vars
        if var.name in self.excluded_vars:
            multiplied_lr_t = lr_t
        else:
            multiplied_lr_t = lr_t * self.lr_mult

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.cast(self._get_hyper("beta_1"), var_dtype)
        beta_2_t = tf.cast(self._get_hyper("beta_2"), var_dtype)
        weight_decay = tf.cast(self._get_hyper("weight_decay"), var_dtype)
        epsilon_t = tf.cast(self.epsilon, var_dtype)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = v.assign(v * beta_2_t)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_corr = m_t / (1.0 - tf.pow(beta_1_t, local_step))
        v_corr = v_t / (1.0 - tf.pow(beta_2_t, local_step))

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = tf.maximum(vhat, v_corr)
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(vhat_t) + epsilon_t) +
                                        multiplied_lr_t * weight_decay * var)
            vhat_update = vhat.assign(vhat_t)
            updates = [var_update, vhat_update]
        else:
            var_update = var.assign_sub(multiplied_lr_t * m_corr /
                                        (tf.sqrt(v_corr) + epsilon_t) +
                                        multiplied_lr_t * weight_decay * var)
            updates = [var_update]

        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate":
            self._serialize_hyperparameter("learning_rate"),
            "beta_1":
            self._serialize_hyperparameter("beta_1"),
            "beta_2":
            self._serialize_hyperparameter("beta_2"),
            "weight_decay":
            self._serialize_hyperparameter("weight_decay"),
            "epsilon":
            self.epsilon,
            "amsgrad":
            self.amsgrad,
            "lr_mult":
            self.lr_mult,
            "excluded_vars":
            self.excluded_vars,
        })
        return config
