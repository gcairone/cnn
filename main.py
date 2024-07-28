import numpy as np
from functions import ReLU, Sigmoid, Softmax
from layers import Linear
from loss import MSELoss, CrossEntropyLoss
from optimizer import SGD, Adam

# Esempio di utilizzo
# activation_function = Sigmoid()  # O Sigmoid(), Softmax(), etc.
optimizer = SGD(learning_rate=0.01)  # O Adam()

layer_1 = Linear(3, 2, activation=Sigmoid(), optimizer=optimizer)
layer_2 = Linear(2, 2, activation=Softmax(), optimizer=optimizer)

# Input fittizio (matrice bidimensionale)
x = np.array([[0.0, 1.0, 2.0]])  # 1 esempio, 3 caratteristiche

# Avanti
hid = layer_1(x)
output = layer_2(hid)
print("Output:", output)

# Target fittizio
y_true = np.array([[1.0, 0.0]])

# Funzione di perdita
loss_function = CrossEntropyLoss()  # O CrossEntropyLoss()

# Calcola la perdita
loss = loss_function(y_true, output)
print("Loss:", loss)

# Calcola il gradiente della perdita rispetto all'output
grad_output = loss_function.grad(y_true, output)
print("Gradient Output:", grad_output)

# Indietro
grad_1 = layer_2.backward(grad_output)
grad_input = layer_1.backward(grad_1)
print("Gradient Input:", grad_input)
