'''
Sampling from a pretrained model
Author: Christian Jacobsen, University of Michigan 2023
'''

import torch
import argparse
import yaml
import os
import os.path as osp
import pickle
from omegaconf import OmegaConf
from utils import instantiate_from_config, plot_inpainting
from pytorch_lightning import seed_everything

# from data.load import PDEloader, TensorDataset
from data.load_test import PDEloader, TensorDataset
from data.transform import PDEtransform
from torch.utils.data import DataLoader

def get_parser():
    parser = argparse.ArgumentParser(description='sampling inputs')
    parser.add_argument('--pde_name', type=str, help='PDE to solve')
    parser.add_argument('--logdir', type=str, help='path to log directory')
    parser.add_argument('--name', type=str, help='name of folder to save samples to')
    parser.add_argument('--config', type=str, help='path to model configuration file [*.yaml]')
    parser.add_argument('--seed', type=int, default=101, help='global seed for reproducibility')
    parser.add_argument('--ckpt', type=str, help='path to model checkpoint (*.ckpt)')
    parser.add_argument('--sensor', type=str, help='path to sensor configuration [*.yaml]')
    return parser

def main():
    parser = get_parser()
    known, unknown = parser.parse_known_args()
    if len(unknown)>0:
        raise Exception('Unknown input argument(s): ', unknown)

    seed_everything(known.seed) # reproducibility

    config = OmegaConf.load(known.config)
    sample_config = OmegaConf.load(known.sensor)

    # sensor placement config, data config, model config
    sensor_config = sample_config.pop("sensors", OmegaConf.create())
    data_config = sample_config.pop("data", OmegaConf.create())
    model_config = config.pop("model", OmegaConf.create())
    model_config.params.sampler_config = sample_config.sampler_config

    # Create folder to save the sampling configuration
    if not osp.exists(known.logdir):
        raise Exception('logdir ', known.logdir, ' does not exist!')
    if not osp.exists(osp.join(known.logdir, known.name)):
        os.makedirs(osp.join(known.logdir, known.name))
    '''
    else:
        raise Exception('sample folder ', osp.join(known.logdir, known.name), ' already exists! Preventing overwrite.')
    '''

    # initialize the model and load checkpoint weights
    model = instantiate_from_config(model_config)
    ckpt = torch.load(known.ckpt)
    model.load_state_dict(ckpt['state_dict'])
    model.to('cuda:'+str(sample_config.device))
    model.eval()

    # initialize sensor function and dataloader
    sensor_fn = instantiate_from_config(sensor_config)
    # datamodule = instantiate_from_config(data_config)
    # datamodule.setup(' ')
    # dataloader = datamodule.train_dataloader()

    data, label = PDEloader(data_config['pde_name']).load_data(data_config['data_dir'])
    data_transform = PDEtransform(data)
    data_tensor = TensorDataset(data_transform.stdtransform(), label)
    dataloader = DataLoader(data_tensor, batch_size=data_config['batch_size'], num_workers=data_config['num_workers'])

    # save sampling configuration
    OmegaConf.save(sensor_config, osp.join(known.logdir, known.name, 'sensor_config.yaml'))
    OmegaConf.save(model_config, osp.join(known.logdir, known.name, 'model_config.yaml'))
    with open(osp.join(known.logdir, known.name, 'cmd_config.yaml'), 'w') as f:
        yaml.dump(known, f)

    batch_size = sample_config.batch_size
    print('batch size: ', batch_size)
    data_batch = next(iter(dataloader))[0][:batch_size]
    # data_batch = next(iter(dataloader))[0][:1].repeat(batch_size, 1, 1, 1) # repeated data
    data_batch = data_batch.to('cuda:'+str(sample_config.device))

    values, indices = sensor_fn(data_batch)
    #indices = []
    sample_batch = model.sample_inpaint(values, indices)

    save_root_path = osp.join(known.logdir, known.pde_name, known.name)
    os.makedirs(save_root_path, exist_ok=True)

    plot_inpainting(data_batch.cpu().detach().numpy(), sample_batch.cpu().detach().numpy(), values.cpu().detach().numpy(), osp.join(known.logdir, known.pde_name, known.name, "samples.png"))

    # compute and compare residuals
    p_l2norm = torch.mean((sample_batch[:, 0, :, :] - data_batch[:, 0, :, :])**2)
    k_l2norm = torch.mean((sample_batch[:, 1, :, :] - data_batch[:, 1, :, :])**2)

    print("L2-norm pressure: ", p_l2norm)
    print("L2-norm permeability: ", k_l2norm)

    save_path = osp.join(known.logdir, known.pde_name, known.name, "results.pkl")

    save_data = {
        "sample_batch": sample_batch.cpu().detach().numpy(),
        "data_batch": data_batch.cpu().detach().numpy(),
        "sol_l2norm": p_l2norm.cpu().detach().numpy(),
        "coef_l2norm": k_l2norm.cpu().detach().numpy()
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    

if __name__ == "__main__":
    main()
    
