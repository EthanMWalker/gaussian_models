import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NTCrossEntropyLoss(nn.Module):
  '''
  NT cross entropy loss
  '''

  def __init__(self, temperature, batch_size, device):
    super().__init__()

    self.temperature = temperature
    self.batch_size = batch_size
    self.device = device
    self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    self.criterion = nn.CrossEntropyLoss(reduction='sum')

  @property
  def negative_representations_mask(self):
    pos_mask = np.zeros(2*self.batch_size)
    pos_mask[[0,self.batch_size]] = 1
    neg_mask = ~pos_mask.astype(bool)
    return torch.from_numpy(neg_mask).to(self.device)
  
  def similarity(self,x,y):
    tmps = []
    for i in range(2*self.batch_size):
      tmp = self.cosine_similarity(x, torch.roll(y,-i,dims=0))
      tmps.append(tmp)
    
    return torch.stack(tmps)
    
  def forward(self, rep1, rep2):
    dbl_batch = 2*self.batch_size

    reps = torch.cat([rep1, rep2], dim=0)

    sims = self.similarity(reps, reps)
    pos_sims = sims[self.batch_size].view(dbl_batch,1)
    neg_sims = sims[self.negative_representations_mask]
    neg_sims = neg_sims.view(dbl_batch, -1)

    logits = torch.cat([pos_sims, neg_sims], dim=1)
    logits /= self.temperature

    labels = torch.zeros(dbl_batch).to(self.device).long()

    loss = self.criterion(logits, labels)
    return loss / dbl_batch