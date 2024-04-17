import pytorch_lightning as pl
import torch
import torch.nn as nn
import ray
from torchmetrics import Accuracy
import utils

#datasets to be used for all models 
vocab = utils.Tokenizer.readVocabulary("SimpleVocab.csv")

#tokenize the data
training, validation, testing = utils.Tokenizer.run_tokenizer(filePath="sequence_iRT_noMods_wihoutBiggestThree.tsv",
                                        vocabPath="SimpleVocab.csv", 
                                            sequenceLength=50,
                                                tokenFormat= utils.TokenFormat.OneDimNoMod)

#make them dataloaders
training = torch.utils.data.DataLoader(training, batch_size=128, shuffle=True, drop_last=True)
validation = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=False, drop_last=True)
testing = torch.utils.data.DataLoader(testing, batch_size=128, shuffle=False, drop_last=True)

#models to test
