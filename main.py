import numpy as np
from dataset import Loader  
from net import SimpleNet  

train_loader = Loader('data/mnist_train.csv', batch_size=16)
test_loader = Loader('data/mnist_test.csv', batch_size=16)

model = SimpleNet()

epochs = 10

def calculate_accuracy(loader, model, max):
    correct = 0
    total = 0
    for x, y in loader:
        outputs = model.forward(x)
        predicted = np.argmax(outputs, axis=1)
        labels = np.argmax(y, axis=1)
        correct += np.sum(predicted == labels)
        total += labels.shape[0]
        if total > max:
            break
    return correct, total

for epoch in range(epochs):
    loss = 0.0
    for x, y in train_loader:
        l = model.backward(x, y)
        loss += l
    
    train_correct, train_total = calculate_accuracy(train_loader, model, 200)
    test_correct, test_total = calculate_accuracy(test_loader, model, 200)
    
    print(f"Epoch: {epoch}, training_loss={loss:.4f}, train_accuracy={train_correct}/{train_total}, test_accuracy={test_correct}/{test_total}")
