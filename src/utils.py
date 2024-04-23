from sklearn.model_selection import train_test_split
from src import tokenize
import numpy as np
import random
import torch
import pandas as pd
import os
from enum import Enum
from PeptidesWithRetentionTimeDataset import PeptidesWithRetentionTimes
import re as regex

class Tokenizer(object):
    '''
    Tokenizer contains all methods related to tokenizing the sequences
    and preparing them for the model.
    '''

    @staticmethod
    def read_psmtsv(filePath: str) -> pd.DataFrame:
        '''
        Reads the psmtsv file and returns a pd dataframe with the 
        Full Sequence and the Scan Reported Retention Time.
        '''

        df = pd.read_csv(filePath, index_col=None,
                            sep="\t",
                            header=0,
                            usecols=["Scan Retention Time",
                                    "Full Sequence"])
        return df

    @staticmethod
    def readVocabulary(filePath: str) -> dict:
        df = pd.read_csv(filePath, index_col="Token")
        vocabularyDictionary = df.to_dict()["Id"]
        return vocabularyDictionary
    
    @staticmethod
    def get_swap_dict(d):
        return {v: k for k, v in d.items()}
    
    @staticmethod
    #Tokenizes the sequence and returns a list of pre-tokenized sequences (clean)
    def getPreTokens(df: pd.DataFrame) -> list:
        preTokens = []
        for index, row in df.iterrows():
            sequence = str(row[0])
            if(sequence.count("[") > 0):
                stars = regex.sub(
                "(?<=[A-HJ-Z])\\[|(?<=\\[)[A-HJ-Z](?=\\])|(?<=[A-HJ-Z])\\](?=$|[A-Z]|(?<=\\])[^A-Z])",
                    "*",
                      sequence)
                removedColon = regex.sub("\\*(.*?):", "*", stars)
                preTokens.append((removedColon, row[1]))
            else:
                preTokens.append((sequence, row[1]))

        return preTokens
    
    @staticmethod
    def tokenizePreTokens(preTokens: list, vocabularyDictionary: dict,
                       sequenceLength: int, tokenFormat: Enum) -> list:
        '''
        Tokenizes the preTokens and returns a list of tokens. 
        Format of each item in the list: [2D np array (sequence, mods), retention time]
        '''

        tokens = []
        # For now will be using the two dimensional format,
        # need to implement the other formats later if necesary
        if(tokenFormat == TokenFormat.TwoDimensional):
            for sequence in preTokens:
                tokenList = []
                modList = []
                #form the tokenList and the modList. 
                #tokenList is the list of tokens for the residues and modList is 
                #the list of tokens for the modifications
                for subSequence in sequence[1].split("*"):
                    if("on" not in subSequence):
                        for residue in subSequence:
                            if(residue in vocabularyDictionary):
                                tokenList.append(vocabularyDictionary[residue])
                                modList.append(0)
                            else:
                                tokenList.clear()
                                modList.clear()
                                break
                    else:
                        if(subSequence in vocabularyDictionary):
                            if(subSequence[len(subSequence)-1] == "X"):
                                tokenList.append(22)
                                modList.append(vocabularyDictionary[subSequence])
                            elif(subSequence[len(subSequence)-1] == "U"):
                                tokenList.append(21)
                                modList.append(0)
                            else:
                                tokenList.append(vocabularyDictionary[subSequence[len(subSequence)-1]])
                                modList.append(vocabularyDictionary[subSequence])
                        else:
                            tokenList.clear()
                            modList.clear()
                            break
                #if the sequence is less than the sequence length, pad it with zeros
                if(len(tokenList) != 0):
                    while(len(tokenList) != sequenceLength and len(modList) < sequenceLength):
                        tokenList.append(0)
                        modList.append(0)
                    #make 2d np array
                    arrayList = []
                    arrayList.append(np.array(tokenList, dtype=np.int32))
                    arrayList.append(np.array(modList, dtype=np.int32))
                    #stack the arrays
                    sequenceWithMods = np.vstack(arrayList)
                    #append the stacked arrays with the retention time to the tokens list
                    tokens.append((sequenceWithMods, float(sequence[0])))

        elif(tokenFormat == TokenFormat.OneDimNoMod):
            for sequence in preTokens:
                tokenList = []
                for residue in sequence[0]:
                    if(residue in vocabularyDictionary):
                        tokenList.append(vocabularyDictionary[residue])
                    else:
                        tokenList.clear()
                        break
                if(len(tokenList) != 0):
                    while(len(tokenList) != sequenceLength):
                        tokenList.append(0)
                    tokens.append((np.array(tokenList, dtype=np.int32), sequence[1]))
        return tokens
    
    @staticmethod
    def run_tokenizer(filePath: str, vocabPath: str,
                       sequenceLength: int, tokenFormat: Enum) -> PeptidesWithRetentionTimes:
        psmtsv_df = Tokenizer.read_psmtsv(filePath)
        vocab = Tokenizer.readVocabulary(vocabPath)
        preTokens = Tokenizer.getPreTokens(psmtsv_df)
        tokens = Tokenizer.tokenizePreTokens(preTokens, vocab, sequenceLength, tokenFormat)
        sequences, retention_times = Tokenizer.prepare_datasets(tokens)
        return PeptidesWithRetentionTimes(sequences, retention_times)

    @staticmethod
    def run_tokenizer(filePath: str, vocabPath: str, sequenceLength: int, tokenFormat: Enum,
                    training_split: float = 0.8, validation_split: float = 0.5,
                      testing_split: float = 0.5, random_state: int = 42) -> PeptidesWithRetentionTimes:
        psmtsv_df = Tokenizer.read_psmtsv(filePath)
        
        #split into training, testing and validation sets
        training_data = psmtsv_df.sample(frac=training_split, random_state=random_state)
        testing_data = psmtsv_df.drop(training_data.index).sample(frac=testing_split, random_state=random_state)
        validation_data = psmtsv_df.drop(training_data.index).drop(testing_data.index)

        #get the vocabulary
        vocab = Tokenizer.readVocabulary(vocabPath)
        
        #get the pretokens
        training_data = Tokenizer.getPreTokens(training_data)
        testing_data = Tokenizer.getPreTokens(testing_data)
        validation_data = Tokenizer.getPreTokens(validation_data)

        #tokenize the dataset
        training_tokens = Tokenizer.tokenizePreTokens(training_data, vocab, sequenceLength, tokenFormat)
        testing_tokens = Tokenizer.tokenizePreTokens(testing_data, vocab, sequenceLength, tokenFormat)
        validation_tokens = Tokenizer.tokenizePreTokens(validation_data, vocab, sequenceLength, tokenFormat)

        #prepare the datasets
        training_sequences, training_retention_times = Tokenizer.prepare_datasets(training_tokens)
        validation_sequences, validation_retention_times = Tokenizer.prepare_datasets(validation_tokens)
        testing_sequences, testing_retention_times = Tokenizer.prepare_datasets(testing_tokens)

        return PeptidesWithRetentionTimes(training_sequences, training_retention_times), \
                PeptidesWithRetentionTimes(validation_sequences, validation_retention_times), \
                PeptidesWithRetentionTimes(testing_sequences, testing_retention_times)
    
    @staticmethod
    def prepare_datasets(tokens: list):
        sequences = []
        retention_times = []
        for i in tokens:
            sequences.append(i[0])
            retention_times.append(i[1])
        return sequences, retention_times
    

class TokenFormat(Enum):
    OneDimensionalRedundant = 1, #mods are in the same dimension as the residues and are redundant(residue,mod) 
    OneDimensionalNonRedundant = 2, #mods are in the same dimension as the residues and are not redundant(modWithResidue) 
    TwoDimensional = 3, #mods are in a separate dimension from the residues
    OneDimNoMod = 4 #no modifications
aa = {"A":1,"C":2,"D":3, "E":4, "F":5, "G":6, "H":7,
                        "I":8, "K":9, "L":10, "M":11, "N":12, "P":13,
                        "Q":14, "R":15, "S":16, "T":17, "V":18, "W":19, "Y":20,
                        "U":21, "X":22} #U is for selenocysteine and X is for any amino acid

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