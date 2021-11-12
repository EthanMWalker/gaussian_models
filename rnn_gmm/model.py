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
    self.mlp = MLP(hidden_dim, hidden_dim*4, 3*n_comps)
    self.n_comps = n_comps
    self.dim = dim
  
  # def log_prob(self, x, means, covs):
  #   prec = torch.rsqrt(F.softplus(covs))
  #   log_p = torch.sum((means*means + x*x - 2*x*means) * (prec**2), dim=1, keepdim=True)
  #   log_det = torch.sum(torch.log(prec), dim=1, keepdim=True)

  #   return -.5 * (self.dim * np.log(2.*np.pi) + log_p) + log_det

  # write a log sum exp with the following things c + log( sum( exp(x_i - c))) where c is reasonable maybe max(x)

  def log_prob(self, x, means, covs, pis):
    covs = F.softplus(covs)
    pis = F.relu(pis)
    pis = torch.transpose(torch.transpose(pis,0,1), torch.sum(pis,dim=1),0,1)
    exp_cov = torch.exp(1./torch.sqrt(covs))
    numer = -.5*(means*means + x*x - 2*x*means) 
    log_p = torch.logsumexp(numer / covs * exp_cov, dim=1, keepdim=True)

    return log_p
  
  def gen_sample(self, N, means, covs, pis):
    covs = F.softplus(covs)
    pis = F.relu(pis)
    pis = torch.transpose(torch.transpose(pis,0,1)*torch.sum(pis,dim=1),0,1)
    sample = torch.normal(means, covs)
    return torch.sum(sample * pis, dim=1).unsqueeze(1)
  

  def sample(self,N, x=None, mask=None):
    if mask is None:
      iters = self.dim
      rest = torch.zeros((N,self.dim*2)).to(self.device)
      zeros = torch.zeros((N,1)).to(self.device)
      concat = torch.cat((zeros, rest), dim=1).unsqueeze(1)
    else:
      iters = int(torch.sum(1-mask).item())
      masks = mask.repeat(N,1)
      x_o = x*mask
      x_o_sq = torch.squeeze(x_o)
      rest = torch.cat((x_o_sq,masks),dim=1)
      zeros = torch.zeros((N,1)).to(self.device)
      concat = torch.cat((zeros,x_o_sq,masks), dim=1).unsqueeze(1)
    
    o, h = self.rnn(concat)
    means, covs, pis = self.mlp(o.squeeze()).chunk(3,dim=1)
    xu = self.gen_sample(N, means, covs, pis)
    sample = xu

    for i in range(1,iters):
      concat = torch.cat((xu,rest), dim=1).unsqueeze(1)
      o, h = self.rnn(concat, h)
      means, covs, pis = self.mlp(o.squeeze()).chunk(3,dim=1)
      xu = self.gen_sample(N, means, covs, pis)
      sample = torch.cat((sample, xu),dim=1)
    
    return sample
  


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
    means, covs, pis = self.mlp(o.squeeze()).chunk(3,dim=1)
    p = self.log_prob(x_u[:,:,0], means, covs, pis)
    

    # find probs for the rest of the states
    for i in range(1,x_u.size(-1)):
      concat = torch.cat((x_u[:,:,i-1],x_o_sq,masks), dim=1).unsqueeze(1)
      o, h = self.rnn(concat, h)
      means, covs = self.mlp(o.squeeze()).chunk(2,dim=1)
      p += self.log_prob(x_u[:,:,i], means, covs, pis)


    return o, p