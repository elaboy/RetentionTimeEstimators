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
from searchGrid import get_datasets, get_training_datasets

def trainTunableRetentionTimeEstimator(config):
    #Data Setup

    # Load the data
    trainingDataset, validationDataset = get_datasets()

    # Create the model
    model = TunableAttentionRegression(input_size = 2707, hidden_size = config["hidden_size"],
                output_size = 1, numberOfHeads = config["numberOfHeads"])
    
    # Create the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # Create the loss function
    criterion = torch.nn.L1Loss()

    # Create the training data loader
    trainingDataLoader = torch.utils.data.DataLoader(trainingDataset,
                                                    batch_size=config["batchSize"],
                                                    shuffle=True, drop_last=True)
    
    validationDataLoader = torch.utils.data.DataLoader(validationDataset,
                                                    batch_size=config["batchSize"],
                                                    shuffle=True, drop_last=True)
    
    # Train the model
    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainingDataLoader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

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
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

       # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(validationDataLoader):
            with torch.no_grad():
                inputs, labels = data

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps),
                    "accuracy": correct / total}, checkpoint=checkpoint,)
    print("Finished Training")

def testBestModel(best_result):
    bestTrainedModel = TunableAttentionRegression(
        best_result.config["hidden_size"], best_result.config["numberOfHeads"])
    checkpointPath = os.path.join(best_result.checkpoint.to_directory(), "checkpoint")

    model_state, optimizer_state = torch.load(checkpointPath)
    bestTrainedModel.load_state_dict(model_state)

    #Load the data
    _, testingDataset = get_training_datasets()
    testingDataLoader = torch.utils.data.DataLoader(testingDataset, batch_size=4,
                                                    shuffle=True, drop_last=True)
    #Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testingDataLoader:
            inputs, labels = data
            outputs = bestTrainedModel(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print("Best trial test set accuracy: {}".format(correct / total))
    
def main(numberOfSamples, maximumNumberOfEpochs):
    config = {
        "hidden_size": tune.sample_from(lambda _: 2 * range(1, 15)),
        "numberOfHeads": tune.sample_from(lambda _: 1 + range(0, 15)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batchSize": tune.choice([4, 8, 16, 32, 64, 128])}
    
    # Set the Ray Tune scheduler
    scheduler = ASHAScheduler(
        max_t=maximumNumberOfEpochs,
        grace_period=1,
        reduction_factor=2)
    
    # set the Ray tuner
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(trainTunableRetentionTimeEstimator),
            resources={"cpu": 16}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=numberOfSamples,
        ),
        param_space=config,)
    
    # Start the Ray Tune search
    results = tuner.fit()

    # Get the best result
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))
    
    # Test the best model
    testBestModel(best_result)

if __name__ == "__main__":
    main(100, 25)