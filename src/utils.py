from sklearn.model_selection import train_test_split

def splitData(data: list, trainSize:float, testSize: float) -> list:
    '''
    Splits the data into training and testing data. Train and test size must add to 1.
    '''
    #split the data into training and testing data
    trainData, testData = train_test_split(data, train_size=trainSize, test_size=testSize)

    return trainData, testData