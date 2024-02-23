from sklearn.model_selection import train_test_split
from src import tokenize
import random

def splitData(data: list, trainSize:float, testSize: float) -> list:
    '''
    Splits the data into training and testing data. Train and test size must add to 1.
    '''
    #split the data into training and testing data
    trainData, testData = train_test_split(data, train_size=trainSize, test_size=testSize)

    return trainData, testData

def get_training_datasets(vocabPath, dataPath, trainSize, validationSize):
    '''
    Returns the training and testing datasets.
    '''
    # Load the vocabulary
    vocab = tokenize.readVocabulary("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\vocab.csv")
    # Load the data
    data = tokenize.readData("C:\\Users\\elabo\\Documents\\GitHub\\RetentionTimeEstimators\\CalibratorTestingMultipleFiles.csv")
    # pre-Tokenize the data
    preTokens = tokenize.getPreTokens(data)
    # Tokenize the data
    tokens = tokenize.tokenizePreTokens(random.sample(preTokens, 1000),
                                        vocab, 100, tokenize.TokenFormat.TwoDimensional)
    train, validate = train_test_split(tokens, train_size=trainSize, test_size=validationSize)

    return trainData, testData