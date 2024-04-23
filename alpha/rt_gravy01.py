import lightning as L
import torch
import torch.nn as nn
import pandas as pd
from src import tokenize
from CustomDatasets.PeptidesWithRetentionTimes import PeptidesWithRetentionTimes
import numpy as np
class AttentionGravy(torch.nn.Module):
    '''
    Convolutional neural network with attention mechanism for retention time prediction.
    '''
    def __init__(self, to_embedd = 222, embedded_dim = 100,
                output_size = 1, numberOfHeads = 10) -> None:
        super().__init__()
        self.embedding = nn.Embedding(to_embedd, embedded_dim, padding_idx = 0)
        self.attention = nn.MultiheadAttention(embedded_dim, num_heads=numberOfHeads)
        self.conv_1 = nn.Conv1d(in_channels=embedded_dim, out_channels=embedded_dim, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=embedded_dim, out_channels=embedded_dim, kernel_size=3, padding=1)
        self.linear = nn.Linear(embedded_dim, 50)
        self.output = nn.Linear(50, output_size)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax()
        self.batch_norm = nn.BatchNorm1d(embedded_dim)
        # self.dropout = nn.Dropout(0.5)
        self.pooling_layer = nn.MaxPool1d(2)

    def forward(self, x): # x = aa, ptm, gravy (100x3)
        #get aa sequence to retrive gravy score
        aa = np.take(x, dim=0)
        
        for index, aa in enumerate(aa):
            if aa in gravy_scores:
                x[index, 2, :] = gravy_scores[aa]
            else:
                x[index, 2, :] = 0.0
        
        #append to x 
        x = torch.cat((x, get_gravy_score(x)), 1)

        #split the input into three tensors 
        aa = x[:, 0, :]
        ptm = x[:, 1, :]
        gravy = x[:, 2, :]

        #embed the input
        embedded_aa = self.embedding(aa)
        embedded_ptm = self.embedding(ptm)
        embedded_gravy = self.embedding(gravy)
        
        #concatenate the embeddings
        embedded_input = torch.cat((embedded_aa, embedded_ptm, embedded_gravy), 1)

        #pass through the attention 
        attention_output, attention_scores = self.attention(embedded_input, embedded_input, embedded_input)

        #pass through the convolutional layers
        conv_output = self.conv_1(attention_output)
        relu_output = self.relu(conv_output)
        pool_output = self.pooling_layer(relu_output)

        conv_output = self.conv_2(pool_output)
        relu_output = self.relu(conv_output)
        pool_output = self.pooling_layer(relu_output)

        #pass through the linear layer
        linear_output = self.linear(pool_output)
        relu_output = self.relu(linear_output)

        return self.output(relu_output)

#pytorch lightning module
class AttentionGravyTrainer(L.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        return optimizer
    
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

def get_gravy_score(sequence: torch.Tensor) -> torch.Tensor:
    '''
    Assigns a gravy score to the sequence based on the average of the gravy scores of the residues in the sequence.
    '''
    gravy_score_index = tokenize.get_swap_dict(gravy_scores_index)

    gravy_score = []

    for residue in sequence:
        if residue in gravy_score_index:
            gravy_score += gravy_scores_index[residue]
    
    return torch.tensor(gravy_score, dtype=torch.float32)

if __name__ == "__main__":
    #parse the HELA dataset
    dataset = pd.read_csv(r"C:\Users\elabo\Documents\MannPeptideResults\Hela_AllPeptides.psmtsv", index_col=None, sep="\t",
                           header=0,
                           usecols=["Scan Retention Time", "Full Sequence"])
    
    #randomly split the dataset into training, testing and validation sets
    training_data = dataset.sample(frac=0.8, random_state=0)
    testing_data = dataset.drop(training_data.index).sample(frac=0.5, random_state=0)
    validation_data = dataset.drop(training_data.index).drop(testing_data.index)

    #create the vocabulary
    vocab = tokenize.readVocabulary("SimpleVocab.csv")

    #get pretokens
    training_data = tokenize.getPreTokens(training_data)
    testing_data = tokenize.getPreTokens(testing_data)
    validation_data = tokenize.getPreTokens(validation_data)

    #tokenize the dataset
    training_tokens = tokenize.tokenizePreTokens(training_data, vocab, 100, tokenize.TokenFormat.TwoDimensional)
    testing_tokens = tokenize.tokenizePreTokens(testing_data, vocab, 100, tokenize.TokenFormat.TwoDimensional)
    validation_tokens = tokenize.tokenizePreTokens(validation_data, vocab, 100, tokenize.TokenFormat.TwoDimensional)

    #create the datasets
    training_sequences = []
    training_retention_times = []
    for i in training_tokens:
        training_sequences.append(i[0])
        training_retention_times.append(i[1])
    
    validation_sequences = []
    validation_retention_times = []
    for i in validation_tokens:
        validation_sequences.append(i[0])
        validation_retention_times.append(i[1])

    testing_sequences = []
    testing_retention_times = []
    for i in testing_tokens:
        testing_sequences.append(i[0])
        testing_retention_times.append(i[1])

    training_dataset = PeptidesWithRetentionTimes(training_sequences, training_retention_times)
    testing_dataset = PeptidesWithRetentionTimes(testing_sequences, testing_retention_times)
    validation_dataset = PeptidesWithRetentionTimes(validation_sequences, validation_retention_times)

    #create the model
    rt_model = AttentionGravy(to_embedd = 222, embedded_dim = 100, output_size = 1, numberOfHeads = 10)
    model = AttentionGravyTrainer(rt_model)
    
    #train the model
    trainer = L.Trainer()
    trainer.fit(model, train_dataloaders = training_dataset, val_dataloaders = validation_dataset)
    trainer.test(model, dataloaders = testing_dataset)




