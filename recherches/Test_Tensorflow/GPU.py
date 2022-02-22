import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


@tf.function(jit_compile=True)
def recompiled_on_launch(a, b):
  return a + b

recompiled_on_launch(tf.ones([1, 10]), tf.ones([1, 10]))
recompiled_on_launch(tf.ones([1, 100]), tf.ones([1, 100]))
