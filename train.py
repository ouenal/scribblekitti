import os
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import ConfusionMatrix

from network.cylinder3d import Cylinder3D
from dataloader.semantickitti import SemanticKITTI
from utils.lovasz import lovasz_softmax
from utils.consistency_loss import PartialConsistencyLoss
from utils.evaluation import compute_iou

class LightningTrainer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._load_dataset_info()
        self.student = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.teacher = Cylinder3D(nclasses=self.nclasses, **config['model'])
        self.initialize_teacher()

        self.loss_ls = lovasz_softmax
        self.loss_cl = PartialConsistencyLoss(H=nn.CrossEntropyLoss, ignore_index=0)

        self.teacher_cm = ConfusionMatrix(self.nclasses)
        self.student_cm = ConfusionMatrix(self.nclasses)
        self.best_miou = 0
        self.best_iou = np.zeros((self.nclasses-1,))

        self.save_hyperparameters('config')

    def forward(self, model, fea, pos):
        output_voxel = model([fea.squeeze(0)], [pos.squeeze(0)], 1)
        return output_voxel[:, :, pos[0,:,0], pos[0,:,1], pos[0,:,2]]

    def training_step(self, batch, batch_idx):
        self.update_teacher()
        student_rpz, student_fea, student_label = batch['student']
        teacher_rpz, teacher_fea, _ = batch['teacher']

        student_output = self(self.student, student_fea, student_rpz)
        teacher_output = self(self.teacher, teacher_fea, teacher_rpz)
        loss = self.loss_cl(student_output, teacher_output, student_label) + \
               self.loss_ls(student_output.softmax(1), student_label, ignore=0)

        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        student_rpz, student_fea, student_label = batch['student']
        teacher_rpz, teacher_fea, teacher_label = batch['teacher']

        student_output = self(self.student, student_fea, student_rpz)
        teacher_output = self(self.teacher, teacher_fea, teacher_rpz)
        loss = self.loss_cl(student_output, teacher_output, student_label) + \
               self.loss_ls(student_output.softmax(1), student_label, ignore=0)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        mask = (teacher_label!=0).squeeze()
        self.student_cm.update(student_output.argmax(1)[:,mask], student_label[:,mask])
        self.teacher_cm.update(teacher_output.argmax(1)[:,mask], teacher_label[:,mask])

    def validation_epoch_end(self, outputs):
        _, student_miou = compute_iou(self.student_cm.compute(), ignore_zero=True)
        self.student_cm.reset()
        self.log('val_student_miou', student_miou, on_epoch=True, prog_bar=True)

        teacher_iou, teacher_miou = compute_iou(self.teacher_cm.compute(), ignore_zero=True)
        self.teacher_cm.reset()
        for class_name, class_iou in zip(self.unique_name, teacher_iou):
            self.log('val_teacher_iou_{}'.format(class_name), class_iou * 100)
        self.log('val_teacher_miou', teacher_miou, on_epoch=True, prog_bar=True)

        if teacher_miou > self.best_miou:
            self.best_miou = teacher_miou
            self.best_iou = np.nan_to_num(teacher_iou) * 100
        self.log('val_best_miou', self.best_miou, on_epoch=True, prog_bar=True)
        self.log('val_best_iou', self.best_iou, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        optimizer = Adam(self.student.parameters(), **self.config['optimizer'])
        return [optimizer]

    def setup(self, stage):
        self.train_dataset = SemanticKITTI(split='train', config=self.config['dataset'])
        self.val_dataset = SemanticKITTI(split='valid', config=self.config['dataset'])

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.config['train_dataloader'])

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, **self.config['val_dataloader'])

    def initialize_teacher(self) -> None:
        self.alpha = 0.99 # TODO: Move to config
        for p in self.teacher.parameters(): p.detach_()

    def update_teacher(self) -> None:
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.data.mul_(alpha).add_(1 - alpha, sp.data)

    def _load_dataset_info(self) -> None:
        dataset_config = self.config['dataset']
        self.nclasses = len(dataset_config['labels'])
        self.unique_label = np.asarray(sorted(list(dataset_config['labels'].keys())))[1:] - 1
        self.unique_name = [dataset_config['labels'][x] for x in self.unique_label + 1]
        self.color_map = torch.zeros(self.nclasses, 3, device='cpu', requires_grad=False)
        for i in range(self.nclasses):
            self.color_map[i,:] = torch.tensor(dataset_config['color_map'][i][::-1], dtype=torch.float32)

    def get_model_callback(self):
        dirpath = os.path.join(self.config['trainer']['default_root_dir'], self.config['logger']['project'])
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=dirpath, filename='{epoch}-{val_teacher_miou:.2f}',
                                                  monitor='val_teacher_miou', mode='max', save_top_k=3)
        return [checkpoint]


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/training.yaml')
    parser.add_argument('--dataset_config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    config =  yaml.safe_load(open(args.config_path, 'r'))
    config['dataset'].update(yaml.safe_load(open(args.dataset_config_path, 'r')))
    wandb_logger = WandbLogger(config=config,
                               save_dir=config['trainer']['default_root_dir'],
                               **config['logger'])
    model = LightningTrainer(config)
    Trainer(logger=wandb_logger,
            callbacks=model.get_model_callback(),
            **config['trainer']).fit(model)