import tensorflow as tf

# Creates some virtual devices (cpu:0, cpu:1, etc.) for using distribution strategy
physical_devices = tf.config.list_physical_devices("CPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration()
    ])

g1 = tf.random.Generator.from_seed(1)
print(g1.normal(shape=[2, 3]))
g2 = tf.random.get_global_generator()
print(g2.normal(shape=[2, 3]))

g1 = tf.random.Generator.from_seed(1, alg='philox')
print(g1.normal(shape=[2, 3]))

g = tf.random.Generator.from_non_deterministic_state()
print(g.normal(shape=[2, 3]))

num_traces = 0


@tf.function
def foo(g):
    global num_traces
    num_traces += 1
    return g.normal([])


foo(tf.random.Generator.from_seed(1))
foo(tf.random.Generator.from_seed(2))
print(num_traces)
