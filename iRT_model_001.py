import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch 
import pytorch_lightning as pl
import utils

# # Load the data
# data = pd.read_excel("41467_2021_23713_MOESM4_ESM.xlsx", sheet_name = "Identifications")
# sequence_iRT = data[["Sequence", "Average iRT"]]
# #rename columns to Full Sequence and Scan Retention Time
# sequence_iRT.columns = ["Full Sequence", "Scan Retention Time"]
# #save as csv
# sequence_iRT.to_csv("sequence_iRT.tsv", index = False, sep="\t")
vocab = utils.Tokenizer.readVocabulary("SimpleVocab.csv")

#tokenize the data
training, validation, testing = utils.Tokenizer.run_tokenizer(filePath="sequence_iRT.tsv",
                                        vocabPath="SimpleVocab.csv", 
                                            sequenceLength=32,
                                                tokenFormat= utils.TokenFormat.OneDimNoMod)

#make them dataloaders
training = torch.utils.data.DataLoader(training, batch_size=32, shuffle=True)
validation = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=False)
testing = torch.utils.data.DataLoader(testing, batch_size=32, shuffle=False, drop_last=True)

epoch = 0
val_epoch = 0
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = nn.functional.pad(x, (0, 4))
        x += residual
        x = self.relu(x)
        return x

#class for the model
class iRT_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), 128)
        self.resnet1 = ResnetBlock(128, 128, 3, 1)
        self.resnet2 = ResnetBlock(128, 128, 3, 1)
        self.resnet3 = ResnetBlock(128, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        # self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 1)
        self.double()

    def forward(self, x):
        x = self.embedding(x)
        # x = x.permute(0, 2, 1)
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        x = self.dropout(x)
        # x = self.flatten(x)
        x = self.linear(x)
        return x

#class for pytorch_lighning module (for training)
class iRT_CNN_PL(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.HuberLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        #take dim 1 and 2
        y_hat = y_hat.squeeze()
        #sum the dim 1 values to end up with 32, 1
        y_hat = torch.sum(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        #take dim 1 and 2
        y_hat = y_hat.squeeze()
        #sum the dim 1 values to end up with 32, 1
        y_hat = torch.sum(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        #take dim 1 and 2
        y_hat = y_hat.squeeze()
        #sum the dim 1 values to end up with 32, 1
        y_hat = torch.sum(y_hat, dim=1)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

if __name__ == "__main__":
    model = iRT_CNN()
    pl_model = iRT_CNN_PL(model)
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu")
    trainer.fit(pl_model, training, validation)
    trainer.test(pl_model, testing)
    torch.save(model.state_dict(), "iRT_model_003_128emb.pth")


    #test the model
    labels = []
    flat_labels = []
    preds = []
    flat_preds = []

    for i, (x, y) in enumerate(testing):
        y_hat = model(x)
        y_hat = y_hat.squeeze()
        y_hat = torch.sum(y_hat, dim=1)
        y_hat = y_hat
        y = y
        preds.append(y_hat.tolist())
        labels.append(y.tolist())
    #flatten the lists
    for sublist in labels:
        for item in sublist:
            flat_labels.append(item)
    for sublist in preds:
        for item in sublist:
            flat_preds.append(item)

    plt.scatter(flat_labels, flat_preds, s=0.1)
    plt.xlabel("True iRT")
    plt.ylabel("Predicted iRT")
    plt.title("True vs Predicted iRT")
    plt.savefig("iRT_model_003_128emb.png")

    print("iRT_model_003_128emb.pth")
    print("Training complete")
