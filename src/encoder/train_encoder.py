import os
from config import cfg
import torch
import numpy as np
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
from pytorch_lightning.loggers import TensorBoardLogger

import dataset
from objective import BrownianBridgeLoss
from model import GPT2Encoder

torch.autograd.set_detect_anomaly(True)

def create_dataloader(dataset, config, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=config['optim_params']['batch_size'],
        shuffle=shuffle,
        pin_memory=True,
        drop_last=shuffle,
        num_workers=config['experiment_params']['data_loader_workers'],
    )
    return loader


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class BrownianBridgeModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._set_dataset()
        self._set_language_encoder()

    def configure_optimizers(self):
        if self.config['optim_params']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config['optim_params']['learning_rate'],
                weight_decay=self.config['optim_params']['decay_factor'])
        elif self.config['optim_params']['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config['optim_params']['learning_rate'],
                momentum=self.config['optim_params']['momentum'])
        return [optimizer], []

    def train_dataloader(self):
        return create_dataloader(self.train_dataset, self.config)

    def test_dataloader(self):
        return create_dataloader(self.test_dataset, self.config, shuffle=False)

    def _set_dataset(self):

        self.train_dataset = dataset.Dataset(
            model_name=self.config['model_params']['model_name'],
            train=True,
            config=self.config['data_params']
        )
        self.test_dataset = dataset.Dataset(
            model_name=self.config['model_params']['model_name'],
            train=False,
            config=self.config['data_params']
        )


    def _set_language_encoder(self):
        self.model = GPT2Encoder(
            hidden_dim=self.config['model_params']['hidden_size'],
            latent_dim=self.config['model_params']['latent_dim'],
            )

        self.model.model.resize_token_embeddings(len(self.train_dataset.tokenizer))


    def forward(self, input_ids, attention_mask):
        feats = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)
        return feats

    def get_feats(self, obs):
        input_ids_i, attention_mask_i = self.train_dataset.tokenize_text(
            obs, device=self.config['experiment_params']['device'])
        input_ids_i = input_ids_i[:, :self.train_dataset.max_length]
        attention_mask_i = attention_mask_i[:, :self.train_dataset.max_length]
        feats_i = self.forward(input_ids=input_ids_i, attention_mask=attention_mask_i)
        return feats_i

    def get_losses_for_batch(self, batch):
        torch.cuda.empty_cache()
        obs_0 = batch['y0']
        obs_t = batch['yt']
        obs_T = batch['yT']
        t_s = batch['t1'].float()
        ts = batch['t2'].float()
        Ts = batch['T'].float()
        feats_0 = self.get_feats(obs_0)
        feats_t = self.get_feats(obs_t)
        feats_T = self.get_feats(obs_T)
        log_q_y_tp1 = self.model.get_log_q(feats_t)
        loss_fn = BrownianBridgeLoss(
            z_0=feats_0,
            z_t=feats_t,
            z_T=feats_T,
            t_=t_s,
            t=ts,
            T=Ts,
            alpha=0,
            var=0,
            log_q_y_T=log_q_y_tp1,
            eps=float(self.config['model_params']['eps']),
            max_seq_len=batch['total_t'].float(),
        )
        loss = loss_fn.get_loss()
        return loss

    def training_step(self, batch):
        loss = self.get_losses_for_batch(batch)
        self.log('train_loss', loss.cpu().detach(), prog_bar=True, on_step=True, sync_dist=True, batch_size=self.config['optim_params']['batch_size'])
        return loss

    def test_step(self, batch):
        loss = self.get_losses_for_batch(batch=batch)
        self.log('test_loss', loss.cpu().detach(), prog_bar=True, on_step=True,sync_dist=True, batch_size=self.config['optim_params']['batch_size'])

        return loss

    def save(self, directory):
        torch.save(self.model.mlp.state_dict(), os.path.join(directory, "mlp.pt"))
        torch.save(self.model.feature_extractor.state_dict(), os.path.join(directory, "feature_extractor.pt"))


def run():

    seed_everything(cfg['experiment_params']['seed'])
    logger = TensorBoardLogger(cfg['experiment_params']['exp_name'], name=cfg['experiment_params']['exp_name'])
    # ckpt_callback = pl.callbacks.ModelCheckpoint(
    #     monitor='eval_loss',
    #     dirpath=cfg['experiment_params']['exp_dir'],
    #     save_top_k=1,
    #     every_n_epochs=cfg['experiment_params']['checkpoint_epochs'],
    # )

    model = BrownianBridgeModel(cfg)
    trainer = pl.Trainer(
        default_root_dir=cfg['experiment_params']['exp_dir'],
        accelerator="gpu", 
        devices=[0],
        max_epochs=int(cfg['experiment_params']['num_epochs']),
        logger=logger,
    )

    # model, train dataloader, test dataloader
    trainer.fit(model)
    ## Save the model
    trainer.save_checkpoint(os.path.join(cfg['experiment_params']['exp_dir'],"{}-{}.ckpt".format('wikisection',cfg["model_params"]["latent_dim"])))
    model.save(directory=cfg['experiment_params']['exp_dir'])
    ## Evaluation:
    trainer.test(model)


if __name__ == "__main__":
    run()