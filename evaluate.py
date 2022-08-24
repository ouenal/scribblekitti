import os
import yaml
import h5py
import argparse
import pathlib
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from train import LightningTrainer

class LightningEvaluator(LightningTrainer):
    def __init__(self, config):
        super().__init__(config)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs)

    def test_dataloader(self):
        return self.val_dataloader()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/distillation.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    parser.add_argument('--ckpt_path', default='output/distillation_original.ckpt')
    args = parser.parse_args()

    config =  yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])

    trainer = Trainer(logger=wandb_logger, **config['trainer'])
    model = LightningEvaluator.load_from_checkpoint(args.ckpt_path, config=config)
    trainer.test(model)