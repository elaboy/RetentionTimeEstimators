import pytorch_lightning as pl
import torch
import torch.nn as nn
import ray
from torchmetrics import Accuracy
import utils

#datasets to be used for all models 
vocab = utils.Tokenizer.readVocabulary("SimpleVocab.csv")

#tokenize the data
training, validation, testing = utils.Tokenizer.run_tokenizer(filePath="sequence_iRT_noMods_wihoutBiggestThree.tsv",
                                        vocabPath="SimpleVocab.csv", 
                                            sequenceLength=50,
                                                tokenFormat= utils.TokenFormat.OneDimNoMod)

#make them dataloaders
training = torch.utils.data.DataLoader(training, batch_size=128, shuffle=True, drop_last=True)
validation = torch.utils.data.DataLoader(validation, batch_size=128, shuffle=False, drop_last=True)
testing = torch.utils.data.DataLoader(testing, batch_size=128, shuffle=False, drop_last=True)

#models to test
class ResNetBlockKaimingNormalFanOut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_out")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_out")
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockKaimingNormalFanIn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in")
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockKaimingUniformFanOut(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_out")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_out")
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockKaimingUniformFanIn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in")
        
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockXavierNormal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockXavierUniform(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockOrthogonal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockSparse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.sparse_(self.conv1.weight, sparsity=0.1)
        nn.init.sparse_(self.conv2.weight, sparsity=0.1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockSparseDense(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.sparse_(self.conv1.weight, sparsity=0.9)
        nn.init.sparse_(self.conv2.weight, sparsity=0.9)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    
class ResNetBlockNormalInit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride)
        self.batchNorm1 = nn.BatchNorm1d(out_channels)
        self.batchNorm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        #init weights
        nn.init.normal_(self.conv1.weight)
        nn.init.normal_(self.conv2.weight)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x

class ResNetBlockRandomInit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
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
        x = utils.padTensorBatchAfterCNN(x, self.in_channels)
        x += residual
        x = self.relu(x)
        return x
    