from ray import train, tune
from ray.tune.search.ax import AxSearch
import pytorch_lightning as pl
import torch
from torch import nn
import ResNetBlocks
from enum import Enum

class DimSwap(nn.Module):
    def __init__(self):
        super(DimSwap, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 1)

class ColapseLastTwoDims(nn.Module):
    def __init__(self):
        super(ColapseLastTwoDims, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), x.size(1)*x.size(2))
    
class LowerDimensionFrom3To2(nn.Module):
    def __init__(self):
        super(LowerDimensionFrom3To2, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), x.size(1)*x.size(2))

class ResNetBlockLayer(object):
    @staticmethod
    def getNKaimingNormalFanOut(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockKaimingNormalFanOut(in_channels=config["embedding_layer_hidden_dim"],
                                                                    out_channels=config["embedding_layer_hidden_dim"],
                                                                    kernel_size=config["kernel_size"],
                                                                    
                                                                    
                                                                    bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))

        return nn.Sequential(*layers)

    @staticmethod
    def getNKaimingNormalFanIn(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockKaimingNormalFanIn(in_channels=config["embedding_layer_hidden_dim"],
                                                                    out_channels=config["embedding_layer_hidden_dim"],
                                                                    kernel_size=config["kernel_size"],
                                                                    
                                                                    
                                                                    bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNKaimingUniformFanOut(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockKaimingUniformFanOut(in_channels=config["embedding_layer_hidden_dim"],
                                                                    out_channels=config["embedding_layer_hidden_dim"],
                                                                    kernel_size=config["kernel_size"],
                                                                    
                                                                    
                                                                    bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNKaimingUniformFanIn(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockKaimingUniformFanIn(in_channels=config["embedding_layer_hidden_dim"],
                                                                    out_channels=config["embedding_layer_hidden_dim"],
                                                                    kernel_size=config["kernel_size"],
                                                                    
                                                                    
                                                                    bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
            
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNXavierNormal(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockXavierNormal(in_channels=config["embedding_layer_hidden_dim"],
                                                                out_channels=config["embedding_layer_hidden_dim"],
                                                                kernel_size=config["kernel_size"],
                                                                
                                                                
                                                                bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)

    @staticmethod
    def getNXavierUniform(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockXavierUniform(in_channels=config["embedding_layer_hidden_dim"],
                                                                    out_channels=config["embedding_layer_hidden_dim"],
                                                                    kernel_size=config["kernel_size"],
                                                                    
                                                                    
                                                                    bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNOrthogonal(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockOrthogonal(in_channels=config["embedding_layer_hidden_dim"],
                                                            out_channels=config["embedding_layer_hidden_dim"],
                                                            kernel_size=config["kernel_size"],
                                                            
                                                            
                                                            bias=config["bias"]))
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNSparse(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockSparse(in_channels=config["embedding_layer_hidden_dim"],
                                                        out_channels=config["embedding_layer_hidden_dim"],
                                                        kernel_size=config["kernel_size"],
                                                        
                                                        
                                                        bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNSparseDense(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockSparseDense(in_channels=config["embedding_layer_hidden_dim"],
                                                            out_channels=config["embedding_layer_hidden_dim"],
                                                            kernel_size=config["kernel_size"],
                                                            
                                                            
                                                            bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNNormalInit(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockNormalInit(in_channels=config["embedding_layer_hidden_dim"],
                                                            out_channels=config["embedding_layer_hidden_dim"],
                                                            kernel_size=config["kernel_size"],
                                                            
                                                            
                                                            bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNRandomInit(config):
        layers = []
        for i in range(config["num_layers"]):
            layers.append(ResNetBlocks.ResNetBlockRandomInit(in_channels=config["embedding_layer_hidden_dim"],
                                                            out_channels=config["embedding_layer_hidden_dim"],
                                                            kernel_size=config["kernel_size"],
                                                            
                                                            bias=config["bias"]))
            layers.append(PaddingLayers.getZeroPadding1d(config))
        
        return nn.Sequential(*layers)

class LinearLayers(object):
    @staticmethod
    def getNLinearLayers(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.Linear(in_features = config["in_features"], 
                                    out_features = config["out_features"],
                                    bias = config["bias"]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

class DroputLayers(object):
    @staticmethod
    def getDropoutLayer(config):
        return nn.Dropout(p = config["p"], inplace = config["inplace"])
    
    @staticmethod
    def getDropout1dLayer(config):
        return nn.Dropout1d(p = config["p"], inplace = config["inplace"])

class NormalizarionLayers(object):
    @staticmethod
    def getBatchNorm1d(config):
        return nn.BatchNorm1d(num_features = config["num_features"],
                            eps = config["eps"],
                            momentum = config["momentum"],
                            affine = config["affine"],
                            track_running_stats = config["track_running_stats"])
    
    @staticmethod
    def getGroupNorm(config):
        return nn.GroupNorm(num_groups = config["num_groups"],
                            num_channels = config["num_channels"],
                            eps = config["eps"],
                            affine = config["affine"])
    
    @staticmethod
    def getSyncBatchNorm(config):
        return nn.SyncBatchNorm(num_features = config["num_features"],
                                eps = config["eps"],
                                momentum = config["momentum"],
                                affine = config["affine"],
                                track_running_stats = config["track_running_stats"])
    
    @staticmethod
    def getLayerNorm(config):
        return nn.LayerNorm(normalized_shape = config["normalized_shape"],
                            eps = config["eps"],
                            elementwise_affine = config["elementwise_affine"])
    
class RecurrentLayers(object):
    @staticmethod    
    def getNLSTM(config):
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
    def getNLSTMCell(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.LSTMCell(input_size = config["input_size"],
                                    hidden_size = config["hidden_size"],
                                    bias = config["bias"]))
        return nn.Sequential(*layers)

    @staticmethod
    def getNRNN(config):
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
    
    @staticmethod
    def getNRNNCell(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.RNNCell(input_size = config["input_size"],
                                    hidden_size = config["hidden_size"],
                                    bias = config["bias"],
                                    nonlinearity = config["nonlinearity"]))
        return nn.Sequential(*layers)
    
    @staticmethod
    def getNGRU(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.GRU(input_size = config["input_size"],
                                hidden_size = config["hidden_size"],
                                num_layers = config["num_layers"],
                                bias = config["bias"],
                                batch_first = config["batch_first"],
                                dropout = config["dropout"],
                                bidirectional = config["bidirectional"]))
        return nn.Sequential(*layers)

    @staticmethod
    def getNGRUCell(config):
        layers = []
        for i in range(config["n_layers"]):
            layers.append(nn.GRUCell(input_size = config["input_size"],
                                    hidden_size = config["hidden_size"],
                                    bias = config["bias"]))
        return nn.Sequential(*layers)

class PaddingLayers(object):
    @staticmethod
    def getZeroPadding1d(config):
        return nn.ZeroPad1d(padding = (0,4))

def getMultiHeadAttention(config):
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

def getMaxPoolLayer(config):
    return nn.MaxPool1d(kernel_size = config["kernel_size"], 
                        stride = config["stride"], 
                        padding = config["padding"], 
                        dilation = config["dilation"], 
                        return_indices = config["return_indices"], 
                        ceil_mode = config["ceil_mode"])

def getAvgPoolLayer(config):
    return nn.AvgPool1d(kernel_size = config["kernel_size"], 
                        stride = config["stride"], 
                        padding = config["padding"], 
                        ceil_mode = config["ceil_mode"], 
                        count_include_pad = config["count_include_pad"])

# this dictionary is going to be used for the search space in ray tune.
# The returned values should be the function output
ResNetTypeInitDict = {
    "KaimingNormalFanOut": ResNetBlockLayer.getNKaimingNormalFanOut,
    "KaimingNormalFanIn": ResNetBlockLayer.getNKaimingNormalFanIn,
    "KaimingUniformFanOut": ResNetBlockLayer.getNKaimingUniformFanOut,
    "KaimingUniformFanIn": ResNetBlockLayer.getNKaimingUniformFanIn,
    "XavierNormal": ResNetBlockLayer.getNXavierNormal, 
    "XavierUniform": ResNetBlockLayer.getNXavierUniform,
    "Orthogonal": ResNetBlockLayer.getNOrthogonal,
    "Sparse": ResNetBlockLayer.getNSparse,
    "SparseDense": ResNetBlockLayer.getNSparseDense,
    "NormalInit": ResNetBlockLayer.getNNormalInit,
    "RandomInit": ResNetBlockLayer.getNRandomInit
}