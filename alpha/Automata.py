import pytorch_lightning as pl
import torch
import lightning as L
import numpy as np
import pandas as pd
from src import tokenize
from CustomDatasets.PeptidesWithRetentionTimes import PeptidesWithRetentionTimes

class Automata():
    '''
    Abstraction of the workflow for prototyping models and testing them.
    '''

    def __init__(self, model: L.LightningModule):
        self.model = model
        self.training_data, self.validation_data, self.testing_data = self.get_datasets()
        self.train_model()

    def get_datasets(self):
        vocab = tokenize.readVocabulary("vocab.csv")
        training_data = pd.read_csv(r"trypsin_train_set.csv", index_col=None, header=0, usecols=["FullSequence", "Mean"])
        testing_data = pd.read_csv(r"trypsin_test_set.csv", index_col=None, header=0, usecols=["FullSequence", "Mean"])
        validation_data = pd.read_csv(r"trypsin_val_set.csv", index_col=None, header=0, usecols=["FullSequence", "Mean"])

        trainingTokens = tokenize.tokenizePreTokens(tokenize.getPreTokens(training_data), vocab, 100, tokenize.TokenFormat.TwoDimensional)
        testingTokens = tokenize.tokenizePreTokens(tokenize.getPreTokens(testing_data), vocab, 100, tokenize.TokenFormat.TwoDimensional)
        validationTokens = tokenize.tokenizePreTokens(tokenize.getPreTokens(validation_data), vocab, 100, tokenize.TokenFormat.TwoDimensional)

        trainingSequences = []
        trainingRetentionTimes = []
        for i in trainingTokens:
            trainingSequences.append(i[0])
            trainingRetentionTimes.append(i[1])

        validationSequences = []
        validationRetentionTimes = []
        for i in validationTokens:
            validationSequences.append(i[0])
            validationRetentionTimes.append(i[1])

        testingSequences = []
        testingRetentionTimes = []
        for i in testingTokens:
            testingSequences.append(i[0])
            testingRetentionTimes.append(i[1])

        trainingDataset = PeptidesWithRetentionTimes(trainingSequences, trainingRetentionTimes)
        testingDataset = PeptidesWithRetentionTimes(testingSequences, testingRetentionTimes)
        validationDataset = PeptidesWithRetentionTimes(validationSequences, validationRetentionTimes)

        return trainingDataset, validationDataset, testingDataset

    def data_loader(self, dataset, shuffle_data: bool = True):
        return torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=shuffle_data, drop_last=True)

    def train_model(self):
        #Pytorch Lightning trainer
        training_data_loader = self.data_loader(self.training_data)
        validation_data_loader = self.data_loader(self.validation_data, False)
        tesing_data_loader = self.data_loader(self.testing_data, False)
        trainer = L.Trainer()
        trainer.fit(self.model, train_dataloaders = training_data_loader, 
                    val_dataloaders = validation_data_loader)
        trainer.test(self.model, dataloaders = tesing_data_loader)

# if __name__ == "__main__":
#     model = Attention_RT()
#     automata = Automata(model)
#     automata.train_model()