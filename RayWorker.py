from ray import train, tune
from ray.tune.search.ax import AxSearch
import pytorch_lightning as pl
import torch
from torch import nn
import ResNetBlocks 
#main objetive is to create a worker class that searches the best hyperparameters using a pytorch lightning model and ray tune
class RayWorker(tune.Trainable):
    def _setup(self, config):
        self.model = train(config)
        self.trainer = pl.Trainer(max_epochs=10)

    def _train(self):
        self.trainer.fit(self.model)
        return self.trainer.callback_metrics

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

def appendEmbeddingLayer(config):
    return nn.Sequential(nn.Embedding(num_embeddings = config["num_embeddings"],
                                      embedding_dim = config["embedding_dim"],
                                      padding_idx = 0, 
                                      scale_grad_by_freq = config["scale_grad_by_freq"], 
                                      sparse = config["sparse"]))
class LinearLayers(object):
    
    @staticmethod
    def appendNLinearLayers(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.Linear(in_features = config["in_features"], 
                                    out_features = config["out_features"],
                                    bias = config["bias"]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

def appendMaxPoolLayer(config):
    return nn.MaxPool1d(kernel_size = config["kernel_size"], 
                        stride = config["stride"], 
                        padding = config["padding"], 
                        dilation = config["dilation"], 
                        return_indices = config["return_indices"], 
                        ceil_mode = config["ceil_mode"])

#todo: add the rest of the blocks 
def appendNResnetLayers(config):
    layers = []
    for i in range(config["n_layers"]):
        layers.append(ResNetBlocks.(config["n_units"]))
    return nn.Sequential(*layers)

class RecurrentLayers(object):
    
    @staticmethod    
    def appendNLSTM(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.LSTM(input_size = config["input_size"],
                                hidden_size = config["hidden_size"],
                                num_layers = config["num_layers"],
                                bias = config["bias"],
                                batch_first = config["batch_first"],
                                dropout = config["dropout"],
                                bidirectional = config["bidirectional"]))
        return nn.Sequential(*layers)

    @staticmethod
    def appendNRNN(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.RNN(input_size = config["input_size"],
                                hidden_size = config["hidden_size"],
                                num_layers = config["num_layers"],
                                nonlinearity = config["nonlinearity"],
                                bias = config["bias"],
                                batch_first = config["batch_first"],
                                dropout = config["dropout"],
                                bidirectional = config["bidirectional"]))
        return nn.Sequential(*layers)

def appendMultiHeadAttention(config):
    layers = []
    for i in range(config["n_layers"]):
        layers.append(nn.MultiheadAttention(embed_dim=config["embed_dim"],
                                            num_heads=config["num_heads"],
                                            dropout=config["dropout"],
                                            bias=config["bias"],
                                            add_bias_kv=config["add_bias_kv"],
                                            add_zero_attn=config["add_zero_attn"],
                                            kdim=config["kdim"],
                                            vdim=config["vdim"],
                                            batch_first=config["batch_first"]))
    return nn.Sequential(*layers)
