import torch
from functools import partial
import os
import sys
import ray
from ray import tune
from ray.train import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from ray import train
from src import tokenize
from src import utils
import tempfile
import numpy
import random
from typing import Dict
from filelock import FileLock
from src.TunableAttentionRegression import TunableAttentionRegression
from CustomDatasets.PeptidesWithRetentionTimes import PeptidesWithRetentionTimes
import sklearn.model_selection
import pandas

# class TunableAttentionRegression(torch.nn.Module):
#     def __init__(self, input_size = 2707, hidden_size = 512,
#                 output_size = 1, numberOfHeads = 1) -> None:
#         super(TunableAttentionRegression, self).__init__()
#         self.embedding = torch.nn.Embedding(input_size, 32)
#         self.lstm = torch.nn.LSTM(32, hidden_size, batch_first=True)
#         self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=numberOfHeads)
#         self.fc = torch.nn.Linear(hidden_size, output_size)
#         self.sigmoid = torch.nn.Sigmoid()

#     def forward(self, x) -> torch.Tensor:
#         embedded = self.embedding(x)
#         lstm_out, _ = self.lstm(embedded)
#         lstm_out = lstm_out.permute(1, 0, 2)  # [seq_len, batch, hidden_size]
#         attention_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
#         output = self.fc(attention_output.mean(dim=0))
#         output = self.sigmoid(output)
#         return output
    
# class PeptidesWithRetentionTimes(torch.utils.data.Dataset):
#     def __init__(self, peptides, retentionTime):
#         self.peptides = peptides
#         self.retentionTimes = retentionTime

#     def __len__(self):
#         return len(self.peptides)

#     def __getitem__(self, index):
#         peptide = self.peptides[index]
#         retentionTime = self.retentionTimes[index]
#         return peptide, retentionTime
    
def get_training_datasets():
    vocab = tokenize.readVocabulary("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\vocab.csv")
    data = tokenize.readData("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\RetentionFileDatasets.csv")
    preTokens = tokenize.getPreTokens(data)
    tokens = tokenize.tokenizePreTokens(random.sample(preTokens, 1000),
                                        vocab, 100, tokenize.TokenFormat.TwoDimensional)
    train, test = utils.splitData(tokens, 0.9, 0.1)

    trainingSequences = []
    trainingRetentionTimes = []
    for i in train:
        trainingSequences.append(i[0])
        trainingRetentionTimes.append(i[1])

    testingSequences = []
    testingRetentionTimes = []
    for i in train:
        testingSequences.append(i[0])
        testingRetentionTimes.append(i[1])

    trainingDataset = PeptidesWithRetentionTimes(trainingSequences, trainingRetentionTimes)
    testingDataset = PeptidesWithRetentionTimes(testingSequences, testingRetentionTimes)

    return trainingDataset, testingDataset

def get_datasets():
    vocab = tokenize.readVocabulary("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\vocab.csv")
    data = tokenize.readData("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\RetentionFileDatasets.csv")
    preTokens = tokenize.getPreTokens(data)
    tokens = tokenize.tokenizePreTokens(random.sample(preTokens, 10000),
                                        vocab, 100, tokenize.TokenFormat.TwoDimensional)
    train, test = utils.splitData(tokens, 0.9, 0.1)

    trainingSequences = []
    trainingRetentionTimes = []
    for i in train:
        trainingSequences.append(i[0])
        trainingRetentionTimes.append(i[1])

    testingSequences = []
    testingRetentionTimes = []
    for i in train:
        testingSequences.append(i[0])
        testingRetentionTimes.append(i[1])

    trainingDataset = PeptidesWithRetentionTimes(trainingSequences, trainingRetentionTimes)
    testingDataset = PeptidesWithRetentionTimes(testingSequences, testingRetentionTimes)

    return trainingDataset, testingDataset

def get_datasets_all():
    '''
    Returns train, validation, and testing datasets (0.8, 0.1, 0.1)
    '''
    vocab = tokenize.readVocabulary("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\vocab.csv")
    data = pandas.read_csv("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\RetentionFileDatasets.csv", index_col=None, header=0)
    preTokens = tokenize.getPreTokens(data)
    tokens = tokenize.tokenizePreTokens(preTokens, vocab, 100, tokenize.TokenFormat.TwoDimensional)
    train, validateAndTest = sklearn.model_selection.train_test_split(tokens, test_size=0.2)
    validate, test = sklearn.model_selection.train_test_split(validateAndTest, test_size=0.5)

    trainingSequences = []
    trainingRetentionTimes = []
    for i in train:
        trainingSequences.append(i[0])
        trainingRetentionTimes.append(i[1])

    validationSequences = []
    validationRetentionTimes = []
    for i in validate:
        validationSequences.append(i[0])
        validationRetentionTimes.append(i[1])

    testingSequences = []
    testingRetentionTimes = []
    for i in train:
        testingSequences.append(i[0])
        testingRetentionTimes.append(i[1])

    trainingDataset = PeptidesWithRetentionTimes(trainingSequences, trainingRetentionTimes)
    testingDataset = PeptidesWithRetentionTimes(testingSequences, testingRetentionTimes)
    validationDataset = PeptidesWithRetentionTimes(validationSequences, validationRetentionTimes)

    return trainingDataset, validationDataset, testingDataset

def train_model(config):
    model = TunableAttentionRegression(config["input_size"],
                                       config["hidden_size"],
                                       config["output_size"],
                                       config["numberOfHeads"])
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # checkpoint = train.get_checkpoint()
    
    # if checkpoint:
    #     checkpoint_state = checkpoint.to_dict()
    #     start_epoch = checkpoint_state["epoch"]
    #     model.load_state_dict(checkpoint_state["net_state_dict"])
    #     optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    # else:
    #     start_epoch = 0
    
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)


    trainSet, testingSet = get_training_datasets()

    trainLoader = torch.utils.data.DataLoader(
           trainSet, batch_size=32, shuffle=True, num_workers=1)
    validationLoader = torch.utils.data.DataLoader(
        testingSet, batch_size=32, shuffle=True, num_workers=1)
    
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainLoader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0]
            labels = data[1]
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0
    
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(validationLoader, 0):
            with torch.no_grad():
                inputs, labels = data
    
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
    
                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
    
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps),
                "accuracy": correct / total},
                checkpoint=checkpoint,)
        # checkpoint_data = {
        #     "epoch": epoch,
        #     "net_state_dict": model.state_dict(),
        #     "optimizer_state_dict": optimizer.state_dict(),
        # }
        # checkpoint = Checkpoint.from_dict(checkpoint_data)
    
        # session.report(
        #     {"loss": val_loss / val_steps, "accuracy": correct / total},
        #     checkpoint=checkpoint,
        # )
    print("Finished Training")
    

def test_accuracy(model, trainset, testset):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    checkpoint_path = os.path.join(model.checkpoint.to_directory(), "checkpoint.pt")


    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            input, labels = data
            outputs = model(input)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main(num_samples=10, max_num_epochs=10):
    config = {
        "input_size": 2707,
        "hidden_size": tune.sample_from(lambda _: 512 * numpy.random.randint(2, 9)),
        "output_size": 1,
        "numberOfHeads": tune.sample_from(lambda _: 2 ** numpy.random.randint(1, 5)),
        "lr": tune.loguniform(1e-4, 1e-1)}
    
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
        resources={"cpu": 4}),
    tune_config = tune.TuneConfig(
        metric="loss",
        mode="min",
        scheduler=scheduler,
        num_samples=num_samples,),
        param_space = config,)
    
    result = tuner.fit()

    # result = tune.run(
    #     partial(train_model, trainingDataset, testingDataset),
    #     resources_per_trial={"cpu": 1},
    #     config=config,
    #     num_samples=num_samples,
    #     scheduler=scheduler)

    best_trial = result.get_best_result("loss", "min", filter_nan_and_inf=True)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = TunableAttentionRegression(best_trial.config["input_size"],
                              best_trial.config["hidden_size"],
                                best_trial.config["output_size"],
                              best_trial.config["numberOfHeads"])
    

    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    # best_checkpoint_data = best_checkpoint.to_dict()

    # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    trainingDataset, testingDataset = get_datasets()

    test_acc = test_accuracy(best_trained_model, trainingDataset, testingDataset)
    print("Best trial test set accuracy: {}".format(test_acc))

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10)