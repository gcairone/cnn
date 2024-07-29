import numpy as np

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
