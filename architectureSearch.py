import utils
import torch 
from torch import nn
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pytorch_lightning as pl
import BuildingBlocks
from BuildingBlocks import ResNetBlockLayer, NormalizarionLayers
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from ray.train.lightning import RayDDPStrategy
from torch.distributed.fsdp import FullStateDictConfig
FullStateDictConfig(offload_to_cpu=True, rank0_only=True)


#define a search space
searchSpace = {
    "num_layers": tune.choice([1, 2, 3, 4, 5, 6, 7, 8 ,9 ,10]),
    "embedding_layer_hidden_dim": tune.choice([32, 64, 128, 256, 512, 1024]), #hidden dimension
    "resnetInitType": tune.choice(
        [ResNetBlockLayer.getNKaimingNormalFanIn, 
        ResNetBlockLayer.getNKaimingNormalFanOut,
        ResNetBlockLayer.getNKaimingUniformFanIn,
        ResNetBlockLayer.getNKaimingUniformFanOut,
        ResNetBlockLayer.getNXavierUniform,
        ResNetBlockLayer.getNXavierNormal,
        ResNetBlockLayer.getNOrthogonal,
        ResNetBlockLayer.getNSparse,
        ResNetBlockLayer.getNNormalInit,
        ResNetBlockLayer.getNRandomInit]),
    "kernel_size" : tune.choice([1,2,3]),
    "stride" : tune.choice([1,2,3]),
    "padding" : tune.choice([1,2,3]),
    "dilation" : tune.choice([1,2,3]),
    "groups" : tune.choice([1,2,3]),
    "bias" : tune.choice([True, False]),
    "normalization_layer" : tune.choice([
        NormalizarionLayers.getBatchNorm1d,
        NormalizarionLayers.getGroupNorm,
        NormalizarionLayers.getSyncBatchNorm,
        NormalizarionLayers.getLayerNorm]),
    "Attention" : tune.choice([True, False]),
    "attention_num_heads" : tune.choice([2, 4, 8, 16, 32]), #number of heads in the attention mechanism
    "attention_dropout" : tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]), #dropout in the attention mechanism
    "dropout" : tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
    "dropout_inplace" : tune.choice([True, False]), #dropout inplace parameter
    "lstm" : tune.choice([True, False]), #use lstm
    "activation" : tune.choice([nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU]),
    "optimizer" : tune.choice([torch.optim.Adam, torch.optim.SGD, torch.optim.AdamW]),
    "learning_rate" : tune.loguniform(1e-4, 1e-1),
    "weight_decay" : tune.loguniform(1e-4, 1e-1),
    "batch_size" : tune.choice([32, 64, 128, 256, 512]),
    "epochs" : tune.choice([10, 20, 30, 40, 50]),
    "loss" : tune.choice([nn.HuberLoss, nn.MSELoss, nn.L1Loss, nn.SmoothL1Loss]),
    "scheduler" : tune.choice([torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.MultiStepLR,
                            torch.optim.lr_scheduler.ExponentialLR,
                            torch.optim.lr_scheduler.CosineAnnealingLR]),
    "scheduler_step_size" : tune.choice([1, 2, 3, 4, 5]),

    }

# Define PyTorch Lightning Module
class Explorer(pl.LightningModule):
    def __init__(self, config):
        super(Explorer, self).__init__()
        self.config = config
        #save model confion from the config file
        self.num_layers = config["num_layers"]
        self.embedding_layer_hidden_dim = config["embedding_layer_hidden_dim"]
        self.resnetInitType = config["resnetInitType"]
        self.kernel_size = config["kernel_size"]
        self.stride = config["stride"]
        self.padding = config["padding"]
        self.dilation = config["dilation"]
        self.groups = config["groups"]
        self.bias = config["bias"]
        self.normalization_layer = config["normalization_layer"]
        self.Attention = config["Attention"]
        self.attention_num_heads = config["attention_num_heads"]
        self.attention_dropout = config["attention_dropout"]
        self.dropout = config["dropout"]
        self.dropout_inplace = config["dropout_inplace"]
        self.lstm = config["lstm"]
        self.activation = config["activation"]
        self.optimizer = config["optimizer"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]
        self.loss = config["loss"]
        self.scheduler = config["scheduler"]
        self.scheduler_step_size = config["scheduler_step_size"]
        self.model = self._build_model()


    def _build_model(self):
        layers = []
        #embedding layer 
        layers.append(nn.Embedding(len(vocab), self.embedding_layer_hidden_dim))

        #if attention is true, add attention layer
        if self.Attention:
            layers.append(nn.MultiheadAttention(self.embedding_layer_hidden_dim, self.attention_num_heads, 
                                                dropout=self.attention_dropout))
        
        #add resnet blocks
        for _ in range(self.num_layers):
            layers.append(self.resnetInitType(self.config))
        
        #dropout layer
        layers.append(nn.Dropout(self.dropout, inplace=self.dropout_inplace))

        #lstm layer
        if self.lstm:
            layers.append(nn.LSTM(self.embedding_layer_hidden_dim, self.embedding_layer_hidden_dim))
        
        #output layer, take last layer size and output 1
        layers.append(nn.Linear(self.embedding_layer_hidden_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y.unsqueeze(1))
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y.unsqueeze(1))
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss(y_hat, y.unsqueeze(1))
        self.log("test_loss", loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.config["optimizer"](self.parameters(), lr=self.config["learning_rate"],
                                              weight_decay=self.config["weight_decay"])
        scheduler = self.config["scheduler"](optimizer, step_size=self.config["scheduler_step_size"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": { "scheduler": scheduler,
                             "monitor": "val_loss"  # Optional: LR scheduler can be tied to a specific metric 
                            }
            }

vocab = utils.Tokenizer.readVocabulary(r"/mnt/f/RetentionTimeProject/SimpleVocab.csv")


# Define the Trainable Function
def train_model(config):
    #datasets to be used for all models 

    #tokenize the data
    training, validation, testing = utils.Tokenizer.run_tokenizer(
        filePath=r"/mnt/f/RetentionTimeProject/sequence_iRT_noMods_wihoutBiggestThree.tsv",
            vocabPath=r"/mnt/f/RetentionTimeProject/SimpleVocab.csv", 
                sequenceLength=50,
                    tokenFormat= utils.TokenFormat.OneDimNoMod)
    #make them dataloaders
    training = torch.utils.data.DataLoader(training, batch_size=32, shuffle=True, drop_last=True)
    validation = torch.utils.data.DataLoader(validation, batch_size=32, shuffle=False, drop_last=True)
    testing = torch.utils.data.DataLoader(testing, batch_size=32, shuffle=False, drop_last=True)
    
    # Initialize PyTorch Lightning module with the config
    model = Explorer(config)
    # strategy = RayDDPStrategy(num_nodes=1, use_gpu=True, num_cpus_per_worker=1)
    plugin = RayLightningEnvironment
    callback = RayTrainReportCallback

    # Define PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy='RayFSDPStrategy',
        # plugins=plugin,
        # callbacks=callback,
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)

    # Train the model

    trainer.fit(model, training, validation)

# Define a function to generate a descriptive name for the directory
def generate_trial_name(trial):
    # Extract relevant model details from the trial config
    num_layers = trial.config["num_layers"]
    embedding_dim = trial.config["embedding_layer_hidden_dim"]
    resnet_init_type = trial.config["resnetInitType"].__name__
    attention = "Attention" if trial.config["Attention"] else "NoAttention"
    lstm = "LSTM" if trial.config["lstm"] else "NoLSTM"
    # Construct a descriptive name
    trial_name = f"NumLayers_{num_layers}_EmbeddingDim_{embedding_dim}_ResnetInit_{resnet_init_type}_{attention}_{lstm}"
    return trial_name


#define the trainable function

#set up ray tune

# Run ray tune experiments

#evaluate results

#shutdown ray??? is that necesary?

if __name__ == "__main__":    
    # Set up Ray Tune
    ray.init()

    from ray.train.torch import TorchTrainer

    scaling_config = ScalingConfig(
    num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1})

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
            ),
        )
    
    ray_trainer = TorchTrainer(train_model,
                                scaling_config=scaling_config,
                                run_config=run_config)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": searchSpace},
        tune_config=tune.TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=20,
        ),
    )

    tuner.fit()


    # Evaluate Results
    # best_trial = tuner.get_best_trial(metric="mean_accuracy", mode="max")
    # best_config = best_trial.config
    # best_accuracy = best_trial.last_result["mean_accuracy"]

    # print("Best configuration:", best_config)
    # print("Best validation accuracy:", best_accuracy)

    # Shut down Ray
    ray.shutdown()