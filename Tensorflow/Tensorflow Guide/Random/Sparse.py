import tensorflow as tf

st1 = tf.sparse.SparseTensor(
    indices=[[0, 3], [2, 4]],
    values=[10, 20],
    dense_shape=[3, 10])

print(st1)


def pprint_sparse_tensor(st):
    s = "<SparseTensor shape=%s \n values={" % (st.dense_shape.numpy().tolist(),)
    for (index, value) in zip(st.indices, st.values):
        s += f"\n  %s: %s" % (index.numpy().tolist(), value.numpy().tolist())
    return s + "}>"


print(pprint_sparse_tensor(st1))

st2 = tf.sparse.from_dense([[1, 0, 0, 8], [0, 0, 0, 0], [0, 0, 3, 0]])
print(pprint_sparse_tensor(st2))

st3 = tf.sparse.to_dense(st2)
print(st3)

st_a = tf.sparse.SparseTensor(indices=[[0, 2], [3, 4]],
                              values=[31, 2],
                              dense_shape=[4, 10])

st_b = tf.sparse.SparseTensor(indices=[[0, 2], [7, 0]],
                              values=[56, 38],
                              dense_shape=[4, 10])

st_sum = tf.sparse.add(st_a, st_b)

print(pprint_sparse_tensor(st_sum))


x = tf.keras.Input(shape=(4,), sparse=True)
y = tf.keras.layers.Dense(4)(x)
model = tf.keras.Model(x, y)

sparse_data = tf.sparse.SparseTensor(
    indices=[(0, 0), (0, 1), (0, 2),
             (4, 3), (5, 0), (5, 1)],
    values=[1, 1, 1, 1, 1, 1],
    dense_shape=(6, 4)
)

model(sparse_data)

model.predict(sparse_data)

dataset = tf.data.Dataset.from_tensor_slices(sparse_data)
for element in dataset:
    print(pprint_sparse_tensor(element))

ragged_x = tf.ragged.constant([["John"], ["a", "big", "dog"], ["my", "cat"]])
ragged_y = tf.ragged.constant([["fell", "asleep"], ["barked"], ["is", "fuzzy"]])
print(tf.concat([ragged_x, ragged_y], axis=1))

sparse_x = ragged_x.to_sparse()
sparse_y = ragged_y.to_sparse()
sparse_result = tf.sparse.concat(sp_inputs=[sparse_x, sparse_y], axis=1)
print(tf.sparse.to_dense(sparse_result, ''))