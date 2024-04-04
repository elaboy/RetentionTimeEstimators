import lightning as L
import torch
from Automata import Automata
import torch.nn as nn


class Conv_RT_Automata(L.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat.squeeze(), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat.squeeze(), y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = torch.nn.functional.mse_loss(y_hat.squeeze(), y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001)
        return optimizer

class Conv_RT(torch.nn.Module):
    '''
    Convolutional neural network with attention mechanism for retention time prediction.
    '''
    def __init__(self, input_size = 2707, hidden_size = 128,
                output_size = 1, numberOfHeads = 16) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_size, 64, padding_idx = 0)
        self.cnn_layer = nn.Conv2d(in_channels=2, out_channels=hidden_size, kernel_size=(3, 3))
        self.attention_layer = nn.Linear(62, 1)
        self.fc_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.double()

    def forward(self, x):
        # Embedding layer
        embedded_input = self.embedding(x)
        
        # CNN layer
        cnn_output = self.relu(self.cnn_layer(embedded_input))
        
        # Attention mechanism
        attention_weights = self.softmax(self.attention_layer(cnn_output))
        attended_features = torch.sum(attention_weights * cnn_output, dim=(2, 3))
        
        # Fully connected layer
        output = self.fc_layer(attended_features)
        return output
    

if __name__ == "__main__":
    #use automata to train the model 
    L.seed_everything(42)
    model = Conv_RT()
    model_pl = Conv_RT_Automata(model)
    automata = Automata(model_pl)
    # trainer = L.Trainer(limit_train_batches=100)
    # trainer.fit(automata)
    # trainer.validate(automata)
    # trainer.test(automata)

