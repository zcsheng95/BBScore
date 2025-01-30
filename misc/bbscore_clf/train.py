from dataset import get_dataloaders
from model import Net

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import RichProgressBar
import wandb, torch, argparse, itertools

import numpy as np

torch.set_float32_matmul_precision("medium")

def get_opts(perm_dom_opt):
    
    if "dom" in perm_dom_opt:
        perm_opt, dom_opt = perm_dom_opt.split("dom_")
        perm_opt = perm_opt + "dom"
        if dom_opt == "all":
            if "local" in perm_opt:
                dom_opt = ["w1", "w2", "w3"]
            elif "perm20" in perm_opt:
                dom_opt = ["enron", "clinton", "yahoo", "yelp"]
            elif "gen" in perm_opt:
                dom_opt = ["neox", "gpt2xl", "llama7b", "llama13b", "llama27b", "llama213b"]
        else:
            dom_opt = [dom_opt]
    else:
        perm_opt = perm_dom_opt
        dom_opt = None
        
    return perm_opt, dom_opt

def get_para_iterator():
    
    lr_list = [0.01, 0.001, 0.0001]
    step_size_list = [5, 10, 20]
    gamma_list = [0.1, 0.5, 0.9]
    patience_list = [5, 10, 20]
    max_epochs_list = [20, 100, 200]
    
    para_iterator = itertools.product(lr_list, step_size_list, gamma_list, patience_list, max_epochs_list)
    
    # new_lr_list = [0.001, 0.0001, 0.00001]
    # new_step_size_list = [5, 10, 20]
    # new_gamma_list = [0.1, 0.5, 0.9]
    # new_patience_list = [5, 10, 20, 200]
    # new_max_epochs_list = [20, 100, 200, 400]
    
    # new_para_iterator = itertools.product(new_lr_list, new_step_size_list, new_gamma_list, new_patience_list, new_max_epochs_list)
    
    # res_para_iterator = set(new_para_iterator) - set(para_iterator)
    
    # return res_para_iterator
    
    return para_iterator

def run_one_test(data_config, lr, step_size, gamma, patience, max_epochs):
    
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data_config)
    
    training_config = {
        "mid_size": 1024,
        "lr": lr,
        "step_size": step_size,
        "gamma": gamma,
        "monitor": "val_loss",
        "mode": "min",
        "patience": patience,
        "log_every_n_steps": 10,
        "max_epochs": max_epochs,
    }

    coh_classifier = Net(training_config)

    wandb_logger = WandbLogger(project="new-bbscore-" + data_config["data_opt"] + "-" + data_config["perm_opt"] + str(data_config["dom_opt"])) if "dom" in data_config["perm_opt"] else WandbLogger(project="new-stocoh-" + data_config["data_opt"] + "-" + data_config["perm_opt"])
    wandb_logger.experiment.config.update(training_config)
    wandb_logger.watch(coh_classifier, log_graph=False)

    early_stop_callback = EarlyStopping(monitor=training_config["monitor"], patience=training_config["patience"], mode=training_config["mode"])

    trainer = pl.Trainer(enable_progress_bar=False, 
                        log_every_n_steps=10, 
                        max_epochs=training_config["max_epochs"], 
                        callbacks=[early_stop_callback],
                        logger=wandb_logger) 

    trainer.fit(model=coh_classifier, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(dataloaders=test_dataloader, ckpt_path="best")

    wandb.finish()
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--perm_dom_opt", type=str, default="gen_dom_llama27b")
    parser.add_argument("--data_opt", type=str, default="wiki")
    args = parser.parse_args()
    
    perm_opt, dom_opt = get_opts(args.perm_dom_opt)
    
    data_config = {
        "perm_opt": perm_opt, 
        "dom_opt": dom_opt,
        "norm_opt": None, 
        "data_opt": args.data_opt, # "wiki" or "gcdc"
    }
    
    para_iterator = get_para_iterator()
    
    for lr, step_size, gamma, patience, max_epochs in para_iterator:
        run_one_test(data_config, lr, step_size, gamma, patience, max_epochs)