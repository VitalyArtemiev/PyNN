import numpy as np

def nonlin(x, deriv=False):
    if (deriv == True):
        return (x * (1 - x))

    return 1 / (1 + np.exp(-x))


def xor(a,b):
    return bool(a != b)


def binfun(a, b, c):
    return xor(a, not(b and not c))


def binfunc( a):
    return binfun(a[0], a[1], a[2])


# the 4th column is for bias term
X = np.array([[0, 0, 0, 1],
              [0, 0, 1, 1],
              [0, 1, 0, 1],
              [0, 1, 1, 1],
              [1, 0, 0, 1],
              [1, 0, 1, 1],
              [1, 1, 0, 1],
              [1, 1, 1, 1]
             ])

# output data
y = np.array([[binfunc(X[0])],
              [binfunc(X[1])],
              [binfunc(X[2])],
              [binfunc(X[3])],
              [binfunc(X[4])],
              [binfunc(X[5])],
              [binfunc(X[6])],
              [binfunc(X[7])]
             ])

print(y)

#np.random.seed(1)


# synapses
syn0 = 2 * np.random.random((4, 8)) - 1  # 4x8 matrix of weights ((3 inputs + 1 bias) x 8 nodes in the hidden layer)
syn1 = 2 * np.random.random((8, 1)) - 1  # 8x1 matrix of weights. (8 nodes x 1 output) - no bias term in the hidden layer.

#main training loop

for j in range(60000):

    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Back propagation of errors using the chain rule.
    l2_error = y - l2
    if (j % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1, deriv=True)

    # update weights (no learning rate term)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output after training")
print(l2)
for i in range(len(l2)):
    print(bool(round(l2[i, 0])),"\t", y[i, 0])


