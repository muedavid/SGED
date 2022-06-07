import tensorflow as tf


def predict_class_postprocessing(prediction, threshold=0.5):

    predictions = tf.math.sigmoid(prediction)
    if isinstance(threshold, list):
        threshold_dim = len(threshold)
        threshold = tf.cast(tf.convert_to_tensor(threshold), tf.float32)
        threshold_mult = 1 / tf.convert_to_tensor(threshold)
        threshold_mult = tf.reshape(threshold_mult, shape=[1, 1, 1, threshold_dim])
    else:
        threshold_mult = 1 / threshold

    value_larger_threshold = tf.where(predictions*threshold_mult >= 1.0, predictions, 0.0)
    max_idx = tf.cast(tf.argmax(value_larger_threshold, axis=-1) + 1, tf.int32)
    predictions = tf.where(tf.reduce_sum(value_larger_threshold, axis=-1) > 0.005, max_idx, 0)

    predictions = tf.expand_dims(predictions, axis=-1)

    return predictions
