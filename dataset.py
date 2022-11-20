import tensorflow as tf
import csv

class Dataset(tf.keras.utils.Sequence):
    def __init__(self, source_path):
        super().__init__()
        self.source_path = source_path
        self.source_data = []

        with open(self.source_path) as f:
            reader = csv.DictReader(f)
            for row in reader:

