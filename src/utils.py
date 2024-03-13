from sklearn.model_selection import train_test_split
from src import tokenize
import random
import torch
import pandas as pd
import os

def splitData(data: list, trainSize:float, testSize: float) -> list:
    '''
    Splits the data into training and testing data. Train and test size must add to 1.
    '''
    #split the data into training and testing data
    trainData, testData = train_test_split(data, train_size=trainSize, test_size=testSize)

    return trainData, testData

def get_training_datasets(vocabPath, dataPath, trainSize, validationSize):
    '''
    Returns the training and testing datasets.
    '''
    # Load the vocabulary
    vocab = tokenize.readVocabulary("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\vocab.csv")
    # Load the data
    data = tokenize.readData("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\CalibratorTestingMultipleFiles.csv")
    # pre-Tokenize the data
    preTokens = tokenize.getPreTokens(data)
    # Tokenize the data
    tokens = tokenize.tokenizePreTokens(random.sample(preTokens, 1000),
                                        vocab, 100, tokenize.TokenFormat.TwoDimensional)
    train, validate = train_test_split(tokens, train_size=trainSize, test_size=validationSize)

    return trainData, testData

from src.tokenize import get_aa_vocab

def read_batched_tensor_get_chronologer_format(dataset: 'TestingTensorsDataset', vocab: dict) -> list:
    '''
    Reads dataset and returns a list of the fullSequence with mass shift from choronologer format in the chronologer_mod_format_dict.
    1-20 in vocab corresponds to the 20 canonical amino acids 
    '''
    full_sequence_with_retention_time = []
    #dataset is a TestingTensorsDataset a child class of torch.utils.data.Dataset
    #iterate through the dataset and get the fullSequence and the mass shift
    for batch in dataset:
        #iterate through the data(batched 2D tensor, batched 1D tensor) and get the fullSequence and the retention time
        tokens = None
        for index, sequence in enumerate(zip(batch[0], batch[1])):
            data = sequence[0]
            rt = sequence[1]
            tokens = (get_tensor_as_tokens(data, vocab))
            #change the tokens to their string representation from the vocab
            fullSequence = []
            #Slice vocab dictionay to get the first 20 tokens which are the residues
            aa = get_aa_vocab()
            for token in tokens:
                if token in chronologer_mod_format_dict: #its a residue or a chronologer mod 
                    fullSequence.append("["+str(chronologer_mod_format_dict[token])+"]")
                elif token in aa: #its a residue
                    fullSequence.append(vocab[token])
            full_sequence_with_retention_time.append(("".join(fullSequence), rt.item()))
    return full_sequence_with_retention_time

def get_tensor_as_tokens(tensor: torch.Tensor, vocab: dict) -> list:
    '''
    Returns a list of tokens id from a tensor.
    '''
    tokens = []
    #tensor is 2D tensor, iterate through the tensor and get tokens. 
    #check each position in both dimensions and get the token, if the second dimension is 0, then it is a residue, so just get the token from vocab
    #else its a modification, get aa from vocab and appends, then get the modification from the dictionary and appends
    tensor_as_list = tensor.tolist()
    for i in range(100):
        if tensor_as_list[1][i] == 0:
            tokens.append(tensor_as_list[0][i])
        else:
            tokens.append(tensor_as_list[0][i])
            tokens.append(tensor_as_list[1][i])
    return tokens

class TestingTensorsDataset(torch.utils.data.Dataset):
   def __init__(self, folder):
       self.files = os.listdir(folder)
       self.folder = folder
   def __len__(self):
       return len(self.files)
   def __getitem__(self, idx):
       return torch.load(f"{self.folder}/{self.files[idx]}")
    

chronologer_mod_format_dict = {
268 : "+57.021464", #carbamidomethyl on M
1001 : "+15.99491", #oxidation on M
2199 : "+79.966331", #phosphorylation on S
2200 : "+79.966331", #phosphorylation on T
2201 : "+79.966331", #phosphorylation on Y
39 : "+42.010565", #acetylation on K
43 : "+42.010565", #acetylation on X
2357 : "+101.023869", #succinylation on K
1108 : "+114.042927", #ubiquitilation on K/GG on K
1965 : "+14.015650", #methylation on K
1969 : "+14.015650", #methylation on R
968 : "+28.031300", #dimethylation on K
971 : "+28.031300", #dimethylation on R
2462 : "+42.046950", #trimethylation on K
2432 : "+224.152478", #TMT on K
2435 : "+224.152478", #TMT on X
2442 : "+229.162932", #TMT10/TMT6plex on K
2445 : "+229.162932", #TMT10/TMT6plex on X
1156 : "-18.010565", #Pyro-glu on E
1134 : "-17.026549", #Pyro-glu on Q
397 : "+39.994915" #Cyclized S-CAM-Cys/ Cys->CamSec on C
}