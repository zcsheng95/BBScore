import torch.nn as nn
import lightning.pytorch as pl
import torch

class pairDiffLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pos_score, neg_score, margin=5.0):
        return torch.mean(torch.relu(margin + pos_score - neg_score))
    
class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        self.dropout_rate = 0.2
        self.in_size = 5
        self.mid_size = config["mid_size"]
        
        self.lr = config["lr"]
        self.step_size = config["step_size"]
        self.gamma = config["gamma"]
        
        fc1_in = self.in_size
        fc1_out = self.mid_size
        fc2_in = fc1_out
        fc2_out = fc2_in // 2
        fc3_in = fc2_out 
        fc3_out = 1
        
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        self.fc2 = nn.Linear(fc2_in, fc2_out)
        self.fc3 = nn.Linear(fc3_in, fc3_out)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
        self.leak_relu = nn.LeakyReLU()
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        
        self.criterion = pairDiffLoss()
        
        self.save_hyperparameters()
        
    def forward(self, x):
        # x = torch.flatten(x, 1) # flatten all dimensions except batch
        
        fc_out = self.fc1(x)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)
        
        fc_out = self.fc2(fc_out)
        fc_out = self.leak_relu(fc_out)
        fc_out = self.dropout_layer(fc_out)
        
        fc_out = self.fc3(fc_out)
        
        return fc_out
    
    def training_step(self, batch, batch_idx):
        pos_feat, neg_feat = batch
        pos_score, neg_score = self(pos_feat), self(neg_feat)
        
        loss = self.criterion(pos_score, neg_score)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        acc = torch.mean((pos_score < neg_score).to(torch.float32))
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pos_feat, neg_feat = batch
        pos_score, neg_score = self(pos_feat), self(neg_feat)
        
        loss = self.criterion(pos_score, neg_score)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        acc = torch.mean((pos_score < neg_score).to(torch.float32))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        pos_feat, neg_feat = batch
        pos_score, neg_score = self(pos_feat), self(neg_feat)
        
        loss = self.criterion(pos_score, neg_score)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        acc = torch.mean((pos_score < neg_score).to(torch.float32))
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
        