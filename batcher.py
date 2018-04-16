import numpy as np

#batcher for getting shuffled training samples
class Batcher:
    def __init__(self, data):
        self.data = np.array(data)
        self.data_size = len(data)
        self.current_index = 0
        self.shuffle()

    def shuffle(self):
        shuffle_indices = np.random.permutation(np.arange(self.data_size))
        self.shuffled_data = self.data[shuffle_indices]

    def get_batch(self, batch_size):
        if self.current_index >= self.data_size:
            self.current_index = 0
            self.shuffle()
        batch_data = self.shuffled_data[self.current_index:self.current_index+batch_size]
        self.current_index += batch_size
        return batch_data
