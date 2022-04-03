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

class LightningTester(LightningTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _load_save_file(self, save_path):
        self.f = h5py.File(save_path, 'w')

    def test_step(self, batch, batch_idx):
        rpz, fea, _ = batch['teacher']
        output = self(self.teacher, fea, rpz)

        conf, pred = torch.max(output.softmax(1), dim=1)
        conf = conf.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        key = os.path.join(self.train_dataset.label_paths[batch_idx])
        conf_key, pred_key = os.path.join(key, 'conf'), os.path.join(key, 'pred')
        self.f.create_dataset(conf_key, data=conf)
        self.f.create_dataset(pred_key, data=pred)

    def test_dataloader(self):
        self.config['train_dataloader']['shuffle'] = False
        self.train_dataset.split = 'test' # Hacky way to prevent augmentation
        return DataLoader(dataset=self.train_dataset, **self.config['train_dataloader'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    parser.add_argument('--checkpoint_path', default='output/training.ckpt')
    parser.add_argument('--save_dir', default='output')
    args = parser.parse_args()

    config =  yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    wandb_logger = WandbLogger(config=config, save_dir=config['trainer']['default_root_dir'], **config['logger'])

    trainer = Trainer(logger=wandb_logger, **config['trainer'])
    model = LightningTester.load_from_checkpoint(args.checkpoint_path, config=config)
    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    model._load_save_file(os.path.join(args.save_dir, 'training_results.h5'))
    trainer.test(model)