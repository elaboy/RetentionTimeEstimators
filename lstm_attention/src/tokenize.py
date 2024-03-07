from enum import Enum
import torch
from src.iguaca_dataset import Iguaca_Dataset

def read_batched_tensor_get_chronologer_format(dataset: 'Iguaca_Dataset', vocab: dict) -> list:
    '''
    Reads dataset and returns a list of the fullSequence with mass shift from choronologer format in the chronologer_mod_format_dict.
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
            for token in tokens:
                if token in chronologer_mod_format_dict: #its a residue or a chronologer mod 
                    fullSequence.append("["+str(chronologer_mod_format_dict[token])+"]")
                elif token in aa: #its a residue
                    fullSequence.append(vocab[token])
            full_sequence_with_retention_time.append(("".join(fullSequence), rt.item()))
    return full_sequence_with_retention_time

def get_tensor_as_tokens(tensor: torch.Tensor) -> list:
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

#Chronologer compatible modifications (https://github.com/searlelab/chronologer)
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

#utils
aa = {"A":1,"C":2,"D":3, "E":4, "F":5, "G":6, "H":7,
                        "I":8, "K":9, "L":10, "M":11, "N":12, "P":13,
                        "Q":14, "R":15, "S":16, "T":17, "V":18, "W":19, "Y":20,
                        "U":21, "X":22} #U is for selenocysteine and X is for any amino acid

class TokenFormat(Enum):
    OneDimensionalRedundant = 1, #mods are in the same dimension as the residues and are redundant(residue,mod) 
    OneDimensionalNonRedundant = 2, #mods are in the same dimension as the residues and are not redundant(modWithResidue) 
    TwoDimensional = 3 #mods are in a separate dimension from the residues

