import utils
import torch 
from torch import nn
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import BuildingBlocks
from BuildingBlocks import ResNetBlockLayer, NormalizarionLayers
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
import os
import tempfile
from ray import train, tune


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
class Explorer(nn.Module):
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
                                                dropout=self.attention_dropout, batch_first=True))
        
        #reshape tensor before passing to resnet blocks
        layers.append(BuildingBlocks.DimSwap())

        #add resnet blocks
        for _ in range(self.num_layers):
            layers.append(self.resnetInitType(self.config))
        
        #collapse last dimensions to end with a 2D tensor


        # #dropout layer
        # layers.append(nn.Dropout(self.dropout, inplace=self.dropout_inplace))

        #lstm layer
        if self.lstm:
            layers.append(nn.LSTM(self.embedding_layer_hidden_dim, self.embedding_layer_hidden_dim))
        
        #output layer, take last layer size and output 1
        layers.append(nn.Linear(self.embedding_layer_hidden_dim*self.embedding_layer_hidden_dim, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
vocab = utils.Tokenizer.readVocabulary(r"/mnt/f/RetentionTimeProject/SimpleVocab.csv")


# Define the Trainable Function
def train_model(config):

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    model = Explorer(config).to(device)

    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    criterion = config['loss']()
    #tokenize the data
    training, validation, testing = utils.Tokenizer.run_tokenizer(
        filePath=r"/mnt/f/RetentionTimeProject/sequence_iRT_noMods_wihoutBiggestThree.tsv",
            vocabPath=r"/mnt/f/RetentionTimeProject/SimpleVocab.csv", 
                sequenceLength=50,
                    tokenFormat= utils.TokenFormat.OneDimNoMod)
    
    batchSize = config['batch_size']

    #make them dataloaders
    training = torch.utils.data.DataLoader(training, batch_size=batchSize, shuffle=True, drop_last=True)
    validation = torch.utils.data.DataLoader(validation, batch_size=batchSize, shuffle=False, drop_last=True)
    testing = torch.utils.data.DataLoader(testing, batch_size=batchSize, shuffle=False, drop_last=True)
    

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
            checkpoint = criterion.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")

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
    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 6, "gpu": 0.25}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=20,
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