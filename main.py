import numpy as np
from dataset import Loader  
from net import SequentialNet  
import layer
from activation import *
from optimizer import *
from loss import *
from eval import calculate_accuracy


train_loader = Loader('data/mnist_train.csv', batch_size=16)
test_loader = Loader('data/mnist_test.csv', batch_size=16)

model = SequentialNet()
model.loss_function = CrossEntropyLoss()
model.layers = [
    layer.Linear(784, 64, activation=ReLU(), optimizer=SGD(learning_rate=0.01)),
    layer.Linear(64, 32, activation=ReLU(), optimizer=SGD(learning_rate=0.01)),
    layer.Linear(32, 10, activation=Softmax(), optimizer=SGD(learning_rate=0.01))
]


epochs = 10


for epoch in range(epochs):
    loss = 0.0
    for x, y in train_loader:
        l = model.backward(x, y)
        loss += l
    
    train_correct, train_total = calculate_accuracy(train_loader, model, 200)
    test_correct, test_total = calculate_accuracy(test_loader, model, 200)
    
    print(f"Epoch: {epoch}, training_loss={loss:.4f}, train_accuracy={train_correct}/{train_total}, test_accuracy={test_correct}/{test_total}")
