import numpy as np
import pickle
import gzip
import mlp
import matplotlib.pyplot as plt

with gzip.open("mnist.pkl.gz", mode="rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")


examples = np.random.randint(40000, size = 10)


# for i in range(len(examples)):
#     image = np.reshape(train_set[0][examples[i], :], [28,28])
#     plt.imshow(image, cmap="gray")
#     plt.show()

train_X = train_set[0]
train_y = train_set[1]
labels_train = np.unique(train_y)
# change y [1D] to Y [2D] sparse array coding class

test_X = test_set[0]
test_y = test_set[1]
labels_test = np.unique(test_y)

def one_hot_encoding(y, labels):
    n_examples = len(y)
    labels = np.unique(y)
    Y = np.zeros((n_examples, len(labels)))
    for index in range(len(labels)):
        positions_index = np.where(y == labels[index])[0]
        Y[positions_index, index] = 1
    return Y

train_Y = one_hot_encoding(train_y, labels_train)
test_Y = one_hot_encoding(test_y, labels_test)

print(train_Y.shape)
print(test_Y.shape)

print(train_X.shape)
print(test_X.shape)
print(test_X.shape)
