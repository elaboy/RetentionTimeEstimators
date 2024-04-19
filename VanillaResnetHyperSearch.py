import utils
import torch
import torch.nn as nn
import ray
from ray import tune, train
import tempfile
from ray.tune.schedulers import ASHAScheduler
import BuildingBlocks
import os

vocab = utils.Tokenizer.readVocabulary(r"/mnt/f/RetentionTimeProject/SimpleVocab.csv")

class ResNetBlockXavierNormal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.double()

        #init weights
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        # x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x = self.batchNorm2(x)
        #x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetiRT(nn.Module):
    def __init__(self, numBlocks, batchSize):
        super().__init__()
        self.numBlocks = numBlocks
        self.batchSize = batchSize
        self.embedding = nn.Embedding(len(vocab), 64, 0)
        self.blocks = nn.ModuleList()
        # self.blocks.append(BuildingBlocks.DimSwap())
        for i in range(numBlocks):
            self.blocks.append(ResNetBlockXavierNormal(64, 64))
        self.outputLayer = nn.Linear(64*64, 1)
        self.double()
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(1)
        x = self.outputLayer(x)
        return x

searchSpace = {
    #sample number of blocks from 1 block to 15 blocks 
    "numBlocks": tune.grid_search(list(range(1, 16)))
}   

#define training model function
def train_model(config):
    vocab = utils.Tokenizer.readVocabulary(r"/mnt/f/RetentionTimeProject/SimpleVocab.csv")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    #define model
    model = ResNetiRT(config['numBlocks'], 32)

    #define loss function
    criterion = nn.MSELoss()
    
    #define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    #tokenize the data
    training, validation, testing = utils.Tokenizer.run_tokenizer(
        filePath=r"/mnt/f/RetentionTimeProject/sequence_iRT_noMods_wihoutBiggestThree.tsv",
            vocabPath=r"/mnt/f/RetentionTimeProject/SimpleVocab.csv", 
                sequenceLength=64,
                    tokenFormat= utils.TokenFormat.OneDimNoMod)
    
    training = torch.utils.data.DataLoader(training, batch_size=32, shuffle=True, drop_last=True)
    validation = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=False, drop_last=True)
    testing = torch.utils.data.DataLoader(testing, batch_size=32, shuffle=False, drop_last=True)


    model.to(device)
    #train the model
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(training):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
        
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(validation, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.cpu().numpy()
                val_steps += 1

    print("Finished Training")

    
if __name__ == "__main__":    
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 1, "gpu": 0.1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=searchSpace,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))