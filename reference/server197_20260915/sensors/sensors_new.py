'''
Define various sampling procedures to sample after training
This includes using predefined sensor measurements with inpainting
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import torch.nn as nn
import numpy as np

class coefRandom(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        indices = torch.zeros((self.sensors*batch_size, 4), dtype=torch.long)
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, self.sensors).view(-1)
        channel_inds = torch.ones((self.sensors, )).repeat(batch_size) # channel 1 = coef
        x0 = torch.arange(dim0)
        x1 = torch.arange(dim1)
        g0, g1 = torch.meshgrid(x0, x1)
        combined = torch.stack((g0, g1), dim=2)
        combined = combined.view(-1, 2)
        dim_inds = torch.randperm(dim0*dim1)[:self.sensors].repeat(batch_size)
        dim0_inds = combined[dim_inds, 0]
        dim1_inds = combined[dim_inds, 1]

        indices[:, 0] = batch_inds
        indices[:, 1] = channel_inds
        indices[:, 2] = dim0_inds
        indices[:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices

class solRandom(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        indices = torch.zeros((self.sensors*batch_size, 4), dtype=torch.long)
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, self.sensors).view(-1)
        channel_inds = torch.zeros((self.sensors, )).repeat(batch_size) # channel 0 = sol
        x0 = torch.arange(dim0)
        x1 = torch.arange(dim1)
        g0, g1 = torch.meshgrid(x0, x1)
        combined = torch.stack((g0, g1), dim=2)
        combined = combined.view(-1, 2)
        dim_inds = torch.randperm(dim0*dim1)[:self.sensors].repeat(batch_size)
        dim0_inds = combined[dim_inds, 0]
        dim1_inds = combined[dim_inds, 1]

        indices[:, 0] = batch_inds
        indices[:, 1] = channel_inds
        indices[:, 2] = dim0_inds
        indices[:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices

class sensorRandom(nn.Module):
    def __init__(self,
                 sensors):
        super().__init__()

        self.sensors = sensors # sensors per sample in batch

    def forward(self, data_batch):
        assert len(data_batch.shape) == 4

        batch_size = data_batch.shape[0]
        channels = data_batch.shape[1]
        dim0 = data_batch.shape[2]
        dim1 = data_batch.shape[3]

        indices = torch.zeros((self.sensors*batch_size*2, 4), dtype=torch.long)

        # solution/end field
        batch_inds = torch.arange(batch_size).view(-1, 1).repeat(1, self.sensors).view(-1)
        channel_inds = torch.zeros((self.sensors, )).repeat(batch_size) # channel 0 = sol
        x0 = torch.arange(dim0)
        x1 = torch.arange(dim1)
        g0, g1 = torch.meshgrid(x0, x1)
        combined = torch.stack((g0, g1), dim=2)
        combined = combined.view(-1, 2)
        dim_inds = torch.randperm(dim0*dim1)[:self.sensors].repeat(batch_size)
        dim0_inds = combined[dim_inds, 0]
        dim1_inds = combined[dim_inds, 1]

        indices[:self.sensors*batch_size, 0] = batch_inds
        indices[:self.sensors*batch_size, 1] = channel_inds
        indices[:self.sensors*batch_size, 2] = dim0_inds
        indices[:self.sensors*batch_size, 3] = dim1_inds

        # coef/initial field
        channel_inds = torch.ones((self.sensors, )).repeat(batch_size) # channel 1 = coef
        indices[self.sensors*batch_size:, 0] = batch_inds
        indices[self.sensors*batch_size:, 1] = channel_inds
        indices[self.sensors*batch_size:, 2] = dim0_inds
        indices[self.sensors*batch_size:, 3] = dim1_inds

        values = torch.zeros_like(data_batch)
        values[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]] = data_batch[indices[:, 0], indices[:, 1], indices[:, 2], indices[:, 3]]

        return values, indices
        
