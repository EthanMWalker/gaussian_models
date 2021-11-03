import torch 
import torch.nn as nn
import torch.distributions as D
from gmm import GaussianMixture
from rnn_gmm.utils import MLP

class RnnGmm(nn.Module):

  def __init__(self, dim, hidden_dim, num_layers, n_comps, device):
    super(RnnGmm, self).__init__()
    self.rnn = nn.RNN(2*dim+1, hidden_dim, num_layers, batch_first=True)
    self.hidden_dim = hidden_dim
    self.device = device
    self.mlp = MLP(dim*hidden_dim, dim*hidden_dim*2, 2*n_comps)
    self.n_comps = n_comps
  
  def gmm(self, means, covs):
    # self.means = torch.randn(n_comps, self.gmm_dim).to(device)
    # self.covs = torch.ones(n_comps, self.gmm_dim).to(device)

    mix = D.Categorical(torch.ones(self.n_comps,).to(self.device))
    comp = D.Independent(D.Normal(means, covs), 1)
    prior = D.MixtureSameFamily(mix, comp)
    return prior

  def forward(self, x, mask):
    N = x.size(0)
    x_o = x*mask
    x_o_sq = torch.sqeeze(x_o)
    masks = mask.repeat(N,1)
    h = torch.zeros(self.hidden_dim)

    bool_mask = torch.gt(1-mask, 0)
    x_u = x[:,:,bool_mask]

    # initial state
    concat = torch.cat(
      (torch.zeros_like(x_u[:,:,0]),x_o_sq,masks), dim=1
    ).unsqueeze(1)
    o, h = self.rnn(concat, h)
    means, covs = self.mlp(o).chunk(2,dim=1)
    p = self.gmm(means,covs).log_prob(x_u[:,:,0])
    

    # find probs for the rest of the states
    for i in range(1,x_u.size(-1)):
      concat = torch.cat((x_u[:,:,i-1],x_o_sq,masks), dim=1).unsqueeze(1)
      o, h = self.rnn(concat, h)
      means, covs = self.mlp(o).chunk(2,dim=1)
      p += self.gmm(means,covs).log_prob(x_u[:,:,i])


    return o, p