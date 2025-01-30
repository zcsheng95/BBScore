import pickle 
import numpy as np
import sys, os, torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class PairwiseCoherenceDataset(Dataset):
    
    def __init__(self, all_scores, transform=None):
        
        self.transform = transform
        
        self.data = {
            "pos_feat": None,
            "neg_feat": None
        }
        
    def __len__(self):
        return self.data["pos_feat"].shape[0]
    
    def __getitem__(self, idx):
        pos_feat = self.data["pos_feat"][idx]
        neg_feat = self.data["neg_feat"][idx]
        
        if self.transform:
            pos_feat, neg_feat = self.transform(pos_feat, neg_feat)
            
        return pos_feat, neg_feat
    
class WikiPair(PairwiseCoherenceDataset):
    
    def __init__(self, all_scores, transform=None, multiplier=False, data_type="train"):
        
        super().__init__(all_scores=all_scores, transform=transform)
        
        pos_feat = []
        neg_feat = []
            
        org_score = np.stack(list(all_scores[data_type].values())).T
        data_len = org_score.shape[0]

        key_list = [key for key in all_scores.keys() if data_type in key and key != data_type]

        for key in key_list:
            if multiplier:
                for i in range(data_len):
                    curr_org_score = org_score[i]
                    
                    curr_score = []
                    for win_size in all_scores[key].keys(): 
                        curr_score.append(all_scores[key][win_size][i])
                    curr_score = np.stack(curr_score).T
                    
                    score_len = curr_score.shape[0]
                    curr_org_score = np.repeat(curr_org_score.reshape(1,-1), repeats=score_len, axis=0)
                    
                    pos_feat.append(curr_org_score)
                    neg_feat.append(curr_score)
            else:
                neg_feat.append(np.stack(list(all_scores[key].values())).T)
                
        self.data["pos_feat"] = np.concatenate(pos_feat) if multiplier else org_score
        self.data["neg_feat"] = np.concatenate(neg_feat)
        
            
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, pos_feat, neg_feat):
        return torch.from_numpy(pos_feat).to(torch.float32), torch.from_numpy(neg_feat).to(torch.float32)
 
def get_dataloaders(data_config):
    
    all_scores = get_scores(data_config)
    
    perm_opt = data_config["perm_opt"]
    
    if "perm20" in perm_opt or "local" in perm_opt or perm_opt[0] == "b":
        mult = True
    else:
        mult = False
    
    
    perm_train_dataset = WikiPair(all_scores, transform=ToTensor(), multiplier=mult, data_type="train")
    perm_test_dataset = WikiPair(all_scores, transform=ToTensor(), multiplier=mult, data_type="test") 

    perm_train_dataset, perm_val_dataset = torch.utils.data.random_split(perm_train_dataset, [0.9, 0.1])

    train_dataloader = DataLoader(perm_train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(perm_val_dataset, batch_size=128, shuffle=False)
    test_dataloader = DataLoader(perm_test_dataset, batch_size=128, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def get_scores(data_config):
    
    perm_opt = data_config["perm_opt"]
    
    if data_config["data_opt"] == "wiki":
        
        score_dir = 'path/to/score_dir'
            
        if perm_opt in ["perm20raw"] or perm_opt[0] == "b" or perm_opt == "local_dom" or perm_opt == "gen_dom":
            filename = "scores_" + perm_opt + ".pkl"
            filepath = os.path.join(score_dir, filename)
            with open(filepath, 'rb') as file: 
                all_scores = pickle.load(file)
                
            # if "perm20" in perm_opt:
            #     filename = "scores_perm.pkl"
            #     filepath = os.path.join(score_dir, filename)
            #     with open(filepath, 'rb') as file: 
            #         all_scores_def = pickle.load(file)
            #     all_scores.pop("perm_test_20")
            #     all_scores["perm_test"] = all_scores_def["perm_test"]

    
        # all_scores = {key: all_scores[key] for key in all_scores.keys() if any([dom in key for dom in dom_opt])}
        
    return all_scores