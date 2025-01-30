#from language_modeling_via_stochastic_processes.src.models.utils import weights_init
import torch
import torch.nn as nn
from transformers import AutoModel

# not train bias
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.zeros_(m.bias)
        m.bias.requires_grad = False



class GPT2Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super(GPT2Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self._init_model()

    def _init_model(self):
        self.model = AutoModel.from_pretrained('gpt2')
        self.model = self.model.eval()
        # turn off all the gradients
        for param in self.model.parameters():
            param.requires_grad = False
        self.mlp = nn.Linear(self.model.wte.embedding_dim, self.hidden_dim)
        self.feature_extractor = self.create_feature_extractor() # data_dim -> hidden_dim
        self.log_q = self.create_log_q()
        self.C_eta = nn.Linear(1, 1)


        self.mlp.apply(weights_init)
        self.feature_extractor.apply(weights_init)
        self.log_q.apply(weights_init)
        self.C_eta.apply(weights_init)

    def create_feature_extractor(self):
        return nn.Sequential(*[
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
                               ])

    def create_log_q(self):
        return nn.Sequential(*[
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Linear(self.latent_dim, 1),
                               ])


    def get_log_q(self, x):
        return self.log_q(x)

    def set_to_train(self):
        pass

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs

    def projection(self, gpt_emb):
        z = self.mlp(gpt_emb) # 32, 100
        z = self.feature_extractor(z)
        return z

    def forward(self, input_ids, attention_mask):
        gpt_emb = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Index into the last hidden state of the sentence (last non-EOS token)
        gpt_emb = self.compute_masked_means(gpt_emb, attention_mask)
        # Albert lang embedding -> feature embedding space
        return self.projection(gpt_emb)
