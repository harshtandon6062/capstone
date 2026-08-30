"""Compatibility loader for the existing trained H5 gesture model."""

import tensorflow as tf


class CompatibleDense(tf.keras.layers.Dense):
    """Dense layer that accepts Keras 3 config in older deserializers."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


def load_gesture_model(model_path):
    """Load the trained model without changing its architecture or weights."""
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"Dense": CompatibleDense},
        compile=False,
    )
