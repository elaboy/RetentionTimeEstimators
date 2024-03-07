import os
import torch

class Iguaca_Dataset(torch.utils.data.Dataset):
   def __init__(self, folder):
       self.files = os.listdir(folder)
       self.folder = folder
   def __len__(self):
       return len(self.files)
   def __getitem__(self, idx):
       return torch.load(f"{self.folder}/{self.files[idx]}")