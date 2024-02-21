from enum import Enum
import string
import pandas
import re as regex
import numpy

#Reads csv file and stores the vocabulary in a dictionary
def readVocabulary(filePath: string) -> dict:
    df = pandas.read_csv(filePath, index_col="Token")
    vocabularyDictionary = df.to_dict()["Id"]
    return vocabularyDictionary

#Read data from a csv file and returns dataframe with full sequence (features) and retention time (target)
def readData(filePath: string) -> pandas.DataFrame:
    df = pandas.read_csv(filePath)
    fullSequenceAndRetentionTime = df[["FullSequence", "Mean"]]
    return fullSequenceAndRetentionTime

#Tokenizes the sequence and returns a list of pre-tokenized sequences (clean)
def getPreTokens(df: pandas.DataFrame) -> list:
    preTokens = []
    for index, row in df.iterrows():
        sequence = str(row[0])
        if(sequence.count("[") > 0):
            stars = regex.sub(
            "(?<=[A-HJ-Z])\\[|(?<=\\[)[A-HJ-Z](?=\\])|(?<=[A-HJ-Z])\\](?=$|[A-Z]|(?<=\\])[^A-Z])",
                "*", sequence)
            removedColon = regex.sub("\\*(.*?):", "*", stars)
            preTokens.append((removedColon, row[1]))
        else:
            preTokens.append((sequence, row[1]))

    return preTokens

def tokenizePreTokens(preTokens: list, vocabularyDictionary: dict,
                       sequenceLength: int, tokenFormat: Enum) -> list:
    '''
    Tokenizes the preTokens and returns a list of tokens. 
    Format of each item in the list: [2D numpy array (sequence, mods), retention time]
    '''

    tokens = []
    #For now will be using the two dimensional format, need to implement the other formats later if necesary
    if(tokenFormat == TokenFormat.TwoDimensional):
        for sequence in preTokens:
            tokenList = []
            modList = []
            #form the tokenList and the modList. 
            #tokenList is the list of tokens for the residues and modList is 
            #the list of tokens for the modifications
            for subSequence in sequence[0].split("*"):
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
                arrayList.append(numpy.array(tokenList))
                arrayList.append(numpy.array(modList))
                #stack the arrays
                sequenceWithMods = numpy.vstack(arrayList)
                #append the stacked arrays with the retention time to the tokens list
                tokens.append((sequenceWithMods, sequence[1]))
                
    return tokens

class TokenFormat(Enum):
    OneDimensionalRedundant = 1, #mods are in the same dimension as the residues and are redundant(residue,mod) 
    OneDimensionalNonRedundant = 2, #mods are in the same dimension as the residues and are not redundant(modWithResidue) 
    TwoDimensional = 3 #mods are in a separate dimension from the residues

#utils
aa = {"A":1,"C":2,"D":3, "E":4, "F":5, "G":6, "H":7,
                        "I":8, "K":9, "L":10, "M":11, "N":12, "P":13,
                        "Q":14, "R":15, "S":16, "T":17, "V":18, "W":19, "Y":20,
                        "U":21, "X":22} #U is for selenocysteine and X is for any amino acid