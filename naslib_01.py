# import naslib
# # Import the search space
# # We begin by importing only the NAS-Bench-201 Search Space
# from naslib.search_spaces import NasBench201SearchSpace
# # Create a new search space object.
# # This object doesn't have an architecture assigned to it yet - it represents the entire search space
# graph = NasBench201SearchSpace(n_classes=10) # CIFAR-10 dataset requires 10 classes

# # Sample a random architecture
# # You can call this method only once on any instance
# graph.sample_random_architecture()
# graph.parse()

# # Get the NASLib representation 
# #(eg: operation index chosen at every edge) of the sampled architecture
# graph.get_hash()
# # This graph is now a NAS-Bench-201 model, which can be used for training
# # Forward pass some dummy data through it to see it in action

# import torch

# x = torch.randn(5, 3, 32, 32) # (Batch_size, Num_channels, Height, Width)

# logits = graph(x)

# print('Shape of the logits:', logits.shape)
# # Import code to convert NASLib graph to the original NAS-Bench-201 representation
# from naslib.search_spaces.nasbench201.conversions import convert_naslib_to_str as convert_naslib_nb201_to_str

# # Get the string representation of this model, that the original authors of NAS-Bench-201 used
# convert_naslib_nb201_to_str(graph)
     
# # Mutating an architecture
# # First, create a new child_graph
# child_graph = NasBench201SearchSpace(n_classes=10)

# # Call mutate on the child graph by passing the parent graph to it
# child_graph.mutate(parent=graph)

# # See the parent and child graph representations. Which edge was mutated?
# print(f'Parent graph: {graph.get_hash()}')
# print(f'Child graph : {child_graph.get_hash()}')
     
# # Now, let's load the queryable tabular NAS-Bench-201 API
# # This API has the training metrics of all the 15625 models in the search space
# # such as train and validation accuracies/losses at every epoch

# # With the NAS-Bench-201 API, we can now query, say, the validation performance of any NB201 model
# # Without it, we would have to train the model from scratch to get this information

# from naslib.utils import get_dataset_api
# benchmark_api = get_dataset_api(search_space='nasbench201', dataset='cifar10')

# # First, import the Metric enum
# from naslib.search_spaces.core import Metric

# # Metric has, among others, these values:
# # Metric.TRAIN_ACCURACY
# # Metric.VAL_ACCURACY
# # Metric.TRAIN_LOSS
# # Metric.TEST_LOSS
# # Metric.TRAIN_TIME

# train_acc_parent = graph.query(metric=Metric.TRAIN_ACCURACY, dataset='cifar10', dataset_api=benchmark_api)
# val_acc_parent = graph.query(metric=Metric.VAL_ACCURACY, dataset='cifar10', dataset_api=benchmark_api)

# print('Performance of parent model')
# print(f'Train accuracy: {train_acc_parent:.2f}%')
# print(f'Validation accuracy: {val_acc_parent:.2f}%')

     
from naslib.defaults.trainer import Trainer
from naslib.search_spaces import NasBench201SearchSpace as NB201

# instantiate the search space object
search_space = NB201()
# import some utilities and parse the configuration file
import logging

from naslib import utils
from naslib.utils import setup_logger, get_dataset_api

# This will read the parameters from the default yaml configuration file, which in this 
# case is located in NASLib/naslib/benchmarks/nas_predictors/discrete_config.yaml.
# You do not have to change this but you can play around with its parameters.
config = utils.get_config_from_args(config_type="nas_predictor")
utils.set_seed(config.seed)
utils.log_args(config)

logger = setup_logger(config.save + "/log.log")
logger.setLevel(logging.INFO)
from naslib.optimizers import RegularizedEvolution as RE

# instantiate the optimizer object using the configuration file parameters
optimizer = RE(config)
# this will load the NAS-Bench-201 data (architectures and their accuracy, runtime, etc).
dataset_api = get_dataset_api(config.search_space, config.dataset)

# adapt the search space to the optimizer type
optimizer.adapt_search_space(search_space, dataset_api=dataset_api)
# since the optimizer has parsed the information of the search space, we do not need to pass the search
# space object to the trainer when instantiating it.
trainer = Trainer(optimizer, config, lightweight_output=False)
# call only a method to run the search for the number of iterations specified in the yaml configuration file.
trainer.search()
# After the search is done, we want to evaluate the test performance of
# the best architecture found using the validation set.
trainer.evaluate(dataset_api=dataset_api)

#save best mdel as pytorch model
trainer.train_top1.save(config.save + '/top1.pt')