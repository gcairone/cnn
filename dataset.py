import pandas as pd
import numpy as np

# Carica i dati dai file CSV
train_df = pd.read_csv('data/mnist_train.csv')
test_df = pd.read_csv('data/mnist_test.csv')

# Separare le etichette dai dati
x_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values
x_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Normalizza i valori dei pixel per essere compresi tra 0 e 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape i dati per aggiungere una dimensione extra per il canale di colore (scala di grigi)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convertire le etichette in matrici di classi binarie (one-hot encoding)
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")
