import pickle
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import time

def set_random_seed(seed):
    torch.manual_seed(seed)


def create_data_on_disk(graph_size, num_samples, is_save=True, filename=None, is_return=False, seed=1234):
    """Generate validation dataset (with SEED) and save
    """

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    set_random_seed(seed)
    depo = torch.rand((num_samples, 2))

    set_random_seed(seed)
    graphs = torch.rand((num_samples, graph_size, 2))

    set_random_seed(seed)
    demand = torch.randint(low=1, high=10, size=(num_samples, graph_size), dtype=torch.float32) / CAPACITIES[graph_size]

    if is_save:
        save_to_pickle('Validation_dataset_{}.pkl'.format(filename), (depo, graphs, demand))

    if is_return:
        return (depo, graphs, demand)


def save_to_pickle(filename, item):
    """Save to pickle
    """
    with open(filename, 'wb') as handle:
        pickle.dump(item, handle)


def read_from_pickle(path, return_data_set=True, num_samples=None):
    """Read dataset from file (pickle)
    """

    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    objects = objects[0]
    if return_data_set:
        depo, graphs, demand = objects
        if num_samples is not None:
            return (depo[:num_samples], graphs[:num_samples], demand[:num_samples])
        else:
            return (depo, graphs, demand)
    else:
        return objects


def generate_data_onfly(num_samples=10000, graph_size=20):
    """Generate temp dataset in memory
    """

    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    depo = torch.rand((num_samples, 2))
    graphs = torch.rand((num_samples, graph_size, 2))
    demand = torch.randint(low=1, high=10, size=(num_samples, graph_size), dtype=torch.float32) / CAPACITIES[graph_size]

    return (depo, graphs, demand)


def get_results(train_loss_results, train_cost_results, val_cost, save_results=True, filename=None, plots=True):

    epochs_num = len(train_loss_results)

    df_train = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                  'loss': train_loss_results,
                                  'cost': train_cost_results,
                                  })
    df_test = pd.DataFrame(data={'epochs': list(range(epochs_num)),
                                 'val_сost': val_cost})
    if save_results:
        df_train.to_excel('train_results_{}.xlsx'.format(filename), index=False)
        df_test.to_excel('test_results_{}.xlsx'.format(filename), index=False)

    if plots:
        plt.figure(figsize=(15, 9))
        ax = sns.lineplot(x='epochs', y='loss', data=df_train, color='salmon', label='train loss')
        ax2 = ax.twinx()
        sns.lineplot(x='epochs', y='cost', data=df_train, color='cornflowerblue', label='train cost', ax=ax2)
        sns.lineplot(x='epochs', y='val_сost', data=df_test, palette='darkblue', label='val cost').set(ylabel='cost')
        ax.legend(loc=(0.75, 0.90), ncol=1)
        ax2.legend(loc=(0.75, 0.95), ncol=2)
        ax.grid(axis='x')
        ax2.grid(True)
        plt.savefig('learning_curve_plot_{}.jpg'.format(filename))
        plt.show()

class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.
        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches
