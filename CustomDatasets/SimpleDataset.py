import torch

class PeptidesWithRetentionTimes(torch.utils.data.Dataset):
    def __init__(self, peptides, retentionTime):
        self.peptides = peptides
        self.retentionTimes = retentionTime

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, index):
        peptide = self.peptides[int(index)]
        retentionTime = self.retentionTimes[int(index)]
        return peptide, retentionTime