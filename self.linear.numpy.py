import numpy as np
import matplotlib.pyplot as plt

# this is a project I have completed in order to understand and practise the way deep learning algorythm work


# 1) forward pass, input_size, output_size
# 2) (get gradient) loss and optimizer
# 3) training loop
#       forward pass
#       get gradient
#       update weights
#      (empty gradients)

# 0)

X = np.array([1, 2, 3, 4], dtype=np.float32)
y = np.array([2, 4, 6, 8], dtype=np.float32)

n_samples, n_features = 4, 1

# 1)
# forward pass

weights = 0.00


def forward(x):
    y_comp = weights*x
    return y_comp


input_size, output_size = n_features, 1


# 2) backward pass

def loss(y_pred, y):
    return ((y_pred - y)**2).mean()

# MSE = 1/N * (w * x - y)**2
# dJ/dw = 1/N 2x (w * x -y)


def optimizer(y, y_pred, x):
    return np.dot(2 * x, y_pred - y).mean()

# 3)
#   prediction before training

print(f"prediction before training:{forward(5)}")

# training loop

learning_rate = 0.01
n_epoch = 20

for epoch in range(n_epoch):

    y_comp = forward(X)
    l = loss(y_comp, y)

    # get the gradients
    dw = optimizer(y, y_comp, X)

    # update weights
    weights -= (learning_rate*dw)

    if epoch % 2 == 0:
        print(f"epoch: {epoch+1}, loss:{l:.8f}")

print(f"current prediction: {forward(5):.5f}")

# plotting the data into a graph in order to visualise the models prediction
predicted = forward(X)
plt.plot(X, y, "ro")
plt.plot(X, predicted, "b")
plt.show()
