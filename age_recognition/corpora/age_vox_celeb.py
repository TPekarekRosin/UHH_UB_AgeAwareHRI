from torch.utils import data
import pandas as pd


# todo setup dataset loading
class AgeVoxCelebDataset(data.Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, engine='python', index_col=False)
        self.n_classes = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data['premise'][idx]
        hypothesis = self.data['hypothesis'][idx]
        label = self.data['label'][idx]
        return premise + hypothesis, label