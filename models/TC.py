import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer
import random


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.BWk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim = 1)
        self.device = device
        self.xentloss = nn.CrossEntropyLoss()
        
  

        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
            
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]


        t_num = random.randint(self.timestep, seq_len - self.timestep)
        #t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps       
        nce = 0  # average over timestep and batch
        nce2 = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        encode_samples = torch.empty((seq_len, batch, self.num_channels)).float().to(self.device)


        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)


        forward_seq = z_aug1[:, :t_samples + 1, :]


        c_t = self.seq_transformer(forward_seq)


        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
       
        afterpred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
            
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
    
        afterpred = self.seq_transformer(pred)
        
        ###backward processing
        


        encode_samples_back = torch.empty((seq_len, batch, self.num_channels)).float().to(self.device)
        backward_seq = z_aug2[:, t_samples - 1 :,:]


        for i in np.arange(0,self.timestep):
            encode_samples_back[i] = z_aug1[:,t_samples - i,:].view(batch,self.num_channels)
        
        pred_back = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        retk = 0
        for i in np.arange(0,self.timestep):
            linear_back = self.Wk[i]
            pred_back[i] = linear_back(afterpred[i])
        for i in np.arange(0,self.timestep):
            t = torch.transpose(pred_back[i],0,1)
            total = torch.mm(encode_samples_back[i],torch.transpose(pred_back[i],0,1))
            #total = torch.mm(encode_samples_back[i], t)
            #l = self.xentloss(encode_samples_back[self.timestep-1-i], pred_back[i])
            nce2 += torch.sum(torch.diag(self.lsoftmax(total)))
            #nce += k
            
        full_seq = z_aug1[:, t_samples - self.timestep: t_samples + self.timestep, :]
        c_f = self.seq_transformer(full_seq)
        
        
        iterpred = self.seq_transformer(pred_back)
        pred2 = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0,self.timestep):
            linear2 = self.Wk[i]
            pred2[i] = linear2(iterpred[i])
        for i in np.arange(0,self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred2[i], 0, 1))
            ##total = torch.mm(pred[i],pred2[i])
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        
        temp = self.seq_transformer(torch.transpose(pred2,0,1))

        
        retk/= self.timestep
        
        nce /= -1. * batch * (self.timestep * 2)
        nce2 /= -1*batch*self.timestep
        return nce, self.projection_head(c_t), nce2, temp