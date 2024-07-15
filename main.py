import numpy as np
from net import SimpleNet
x_train = np.random.randn(100, 1, 28, 28)  
y_train = np.random.randint(0, 10, size=100)  

net = SimpleNet()

net.train(x_train, y_train, epochs=10, learning_rate=0.001)

x_test = np.random.randn(1, 1, 28, 28)  
prediction = net.predict(x_test)
print("Prediction:", prediction)
