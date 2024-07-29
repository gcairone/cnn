import pandas as pd
import numpy as np

class Loader:
    def __init__(self, data_path, batch_size=32, shuffle=True):
        self.data = pd.read_csv(data_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = 0
        self.num_samples = self.data.shape[0]
        self.num_batches = self.num_samples // self.batch_size
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def __iter__(self):
        self.index = 0
        if self.shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        return self

    def __next__(self):
        if self.index >= self.num_samples:
            raise StopIteration

        batch_data = self.data.iloc[self.index:self.index + self.batch_size]
        self.index += self.batch_size

        x_batch = batch_data.drop('label', axis=1).values
        y_batch = batch_data['label'].values

        x_batch = x_batch.astype('float32') / 255.0
        x_batch = x_batch.reshape(-1, 784)

        y_batch = np.eye(10)[y_batch]

        return x_batch, y_batch
