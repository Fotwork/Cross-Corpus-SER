from torch.utils.data import Dataset
import torch

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_values = self.data[idx]['input_values']
        label = self.data[idx]['label']
        language_id = self.data[idx]['language_id']

        input_values = torch.tensor(input_values, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        language_id = torch.tensor(language_id, dtype=torch.long)

        return input_values, label, language_id