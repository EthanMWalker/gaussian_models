import torch 
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

import numpy as np

from gmm import GaussianMixture
from rnn_gmm.utils import MLP

class RnnGmm(nn.Module):

  def __init__(self, dim, hidden_dim, num_layers, n_comps, device):
    super(RnnGmm, self).__init__()
    self.rnn = nn.RNN(2*dim+1, hidden_dim, num_layers, batch_first=True)
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.device = device
    self.mlp = MLP(hidden_dim, hidden_dim*4, 2*n_comps)
    self.n_comps = n_comps
    self.dim = dim
  
  # def gmm(self, means, covs):
  #   # self.means = torch.randn(n_comps, self.gmm_dim).to(device)
  #   # self.covs = torch.ones(n_comps, self.gmm_dim).to(device)
  #   covs = torch.abs(covs) + 1.
  #   mix = D.Categorical(torch.ones(self.n_comps,).to(self.device))
  #   norm = D.Normal(means, covs)
  #   comp = D.Independent(norm, 1)
  #   prior = D.MixtureSameFamily(mix, comp)
  #   return prior
  
  def log_prob(self, x, means, covs):
    prec = torch.rsqrt(F.softplus(covs))
    log_p = torch.sum((means*means + x*x - 2*x*means) / (prec**2), dim=1, keepdim=True)
    log_det = torch.sum(torch.log(prec), dim=1, keepdim=True)

    return -.5 * (self.dim * np.log(2.*np.pi) + log_p) + log_det



  def forward(self, x, mask):
    N = x.size(0)
    x = x.unsqueeze(1)
    x_o = x*mask
    x_o_sq = torch.squeeze(x_o)
    masks = mask.repeat(N,1)

    bool_mask = torch.gt(1-mask, 0)
    x_u = x[:,:,bool_mask]

    p = 0.

    # initial state
    concat = torch.cat(
      (torch.zeros_like(x_u[:,:,0]),x_o_sq,masks), dim=1
    ).unsqueeze(1)
    o, h = self.rnn(concat)
    means, covs = self.mlp(o.squeeze()).chunk(2,dim=1)
    p = self.log_prob(x_u[:,:,0], means, covs)
    

    # find probs for the rest of the states
    for i in range(1,x_u.size(-1)):
      concat = torch.cat((x_u[:,:,i-1],x_o_sq,masks), dim=1).unsqueeze(1)
      o, h = self.rnn(concat, h)
      means, covs = self.mlp(o.squeeze()).chunk(2,dim=1)
      p += self.log_prob(x_u[:,:,i], means, covs)


    return o, p