from enum import Enum
import string
import pandas as pd
import re as regex
import numpy
import torch

class PeptidesWithRetentionTimes(torch.utils.data.Dataset):
    def __init__(self, peptides, retentionTime):
        self.peptides = peptides
        self.retentionTimes = retentionTime

    def __len__(self):
        return len(self.peptides)

    def __getitem__(self, index):
        peptide = self.peptides[index]
        retentionTime = self.retentionTimes[index]
        return peptide, retentionTime

class Tokenizer(object):
    '''
    Tokenizer contains all methods related to tokenizing the sequences
    and preparing them for the model.
    '''

    @staticmethod
    def read_psmtsv(filePath: string) -> pd.DataFrame:
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
    def readVocabulary(filePath: string) -> dict:
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
        Format of each item in the list: [2D numpy array (sequence, mods), retention time]
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
                    #make 2d numpy array
                    arrayList = []
                    arrayList.append(numpy.array(tokenList, dtype=numpy.int32))
                    arrayList.append(numpy.array(modList, dtype=numpy.int32))
                    #stack the arrays
                    sequenceWithMods = numpy.vstack(arrayList)
                    #append the stacked arrays with the retention time to the tokens list
                    tokens.append((sequenceWithMods, float(sequence[0])))
                    
        return tokens
    
    @staticmethod
    def get_gravy_score(sequence: torch.Tensor) -> torch.Tensor:
        '''
        Assigns a gravy score to the sequence based on the average of the gravy scores of the residues in the sequence.
        '''
        gravy_score_index = Tokenizer.get_swap_dict(gravy_scores_index)

        gravy_score = []

        for residue in sequence:
            if residue in gravy_score_index:
                gravy_score += gravy_scores_index[residue]
        
        return torch.tensor(gravy_score, dtype=torch.float32)
    
    @staticmethod
    def run_tokenizer(filePath: string, vocabPath: string, sequenceLength: int, tokenFormat: Enum) -> PeptidesWithRetentionTimes:
        psmtsv_df = Tokenizer.read_psmtsv(filePath)
        vocab = Tokenizer.readVocabulary(vocabPath)
        preTokens = Tokenizer.getPreTokens(psmtsv_df)
        tokens = Tokenizer.tokenizePreTokens(preTokens, vocab, sequenceLength, tokenFormat)
        sequences, retention_times = Tokenizer.prepare_datasets(tokens)
        return PeptidesWithRetentionTimes(sequences, retention_times)

    @staticmethod
    def run_tokenizer(filePath: string, vocabPath: string, sequenceLength: int, tokenFormat: Enum,
                    training_split: float = 0.8, validation_split: float = 0.5, testing_split: float = 0.5, random_state: int = 42) -> PeptidesWithRetentionTimes:
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

        return PeptidesWithRetentionTimes(training_sequences, training_retention_times), PeptidesWithRetentionTimes(validation_sequences, validation_retention_times), PeptidesWithRetentionTimes(testing_sequences, testing_retention_times)
    
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
    TwoDimensional = 3 #mods are in a separate dimension from the residues

aa = {"A":1,"C":2,"D":3, "E":4, "F":5, "G":6, "H":7,
                        "I":8, "K":9, "L":10, "M":11, "N":12, "P":13,
                        "Q":14, "R":15, "S":16, "T":17, "V":18, "W":19, "Y":20,
                        "U":21, "X":22} #U is for selenocysteine and X is for any amino acid

gravy_scores = {
    "A" : 1.800,
    "R" : -4.500,
    "N" : -3.500,
    "D" : -3.500,
    "C" : 2.500,
    "Q" : -3.500,
    "E" : -3.500,
    "G" : -0.400,
    "H" : -3.200,
    "I" : 4.500,
    "L" : 3.800,
    "K" : -3.900,
    "M" : 1.900,
    "F" : 2.800,
    "P" : -1.600,
    "S" : -0.800,
    "T" : -0.700,
    "W" : -0.900,
    "Y" : -1.300,
    "V" : 4.200,
    }

gravy_scores_index = {
    1 : 1.800,
    15 : -4.500,
    12 : -3.500,
    3 : -3.500,
    2 : 2.500,
    14 : -3.500,
    4 : -3.500,
    6 : -0.400,
    7 : -3.200,
    8 : 4.500,
    10 : 3.800,
    9 : -3.900,
    11 : 1.900,
    5 : 2.800,
    13 : -1.600,
    16 : -0.800,
    17 : -0.700,
    19 : -0.900,
    20 : -1.300,
    18 : 4.200,
    }