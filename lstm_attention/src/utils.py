import torch 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def save_dataset(dataset: torch.utils.data.Dataset, path: str, verbose = True) -> None:
    '''
    Saves the dataset to a file.
    '''
    torch.save(dataset, path)
    
    if verbose:
        print(f"Dataset saved to: {path}")

def load_dataset(path: str, verbose=True) -> torch.utils.data.dataset:
    '''
    Loads the dataset from a file.
    '''
    dataset = torch.load(path)
    
    if verbose:
        print(f"Dataset loaded from: {path}")
        #print dataset length and shape of the first item in the dataset
        print(f"Length of dataset: {len(dataset)}")
        print(f"Shape of first item in dataset: {dataset[0][0].shape}")

    return dataset

def save_dataLoader(dataLoader: torch.utils.data.DataLoader, path: str, verbose = True) -> None:
    '''
    Saves the dataLoader to a file.
    '''
    torch.save(dataLoader, path)
    
    if verbose:
        print(f"DataLoader saved to: {path}")

def load_dataLoader(path: str, verbose=True) -> torch.utils.data.DataLoader:
    '''
    Loads the dataLoader from a file.
    '''
    dataLoader = torch.load(path)
    
    if verbose:
        print(f"DataLoader loaded from: {path}")
        #print dataset length and shape of the first item in the dataset
        print(f"Length of DataLoader: {len(dataLoader)}")
        print(f"Shape of first item in DataLoader: {dataLoader.dataset[0][0].shape}")

    return dataLoader

def get_swap_dict(d: dict) -> dict:
    '''
    Returns a dictionary with the keys and values swapped.
    '''
    return {v: k for k, v in d.items()}

def scatter_save(data: list, path: str, show = False) -> None:
    '''
    Scatter plot of predictions vs actual values.

    Parameters:
    -----------
    - data: list of tuples (label (torch.Tensor), prediction(torch.Tensor))
    - path: path to save the scatter plot e.g. (path/to/scatter_plot.png)
    - show: if True, the scatter plot will be displayed.
    '''

    flat_labels = []
    flat_predictions = []

    for label, prediction in data:
        flat_predictions.append(prediction)
        flat_labels.append(label)
    
    stacked_flat_labels = torch.stack(flat_labels).flatten().tolist()
    stacked_flat_predictions = torch.stack(flat_predictions).flatten().tolist()
    std = torch.std(stacked_flat_labels - torch.tensor(stacked_flat_predictions)).numpy()
    n = len(stacked_flat_labels)
    r2 = r2_score(stacked_flat_labels, stacked_flat_predictions)
    #scatter plot
    plt.scatter(stacked_flat_labels, stacked_flat_predictions, s=0.1)
    #labels vs labels
    plt.plot(stacked_flat_labels, stacked_flat_labels, color='red')
    plt.xlabel('Labels')
    plt.ylabel('Predictions')
    plt.title('Retention time predictions vs retention times reported in the scan')
    plt.text(0.05, 0.95, f'R2: {r2:.2f}, STD: {std:.2f}, N: {n}', )
    plt.savefig(path)

    if show:
        plt.show()

