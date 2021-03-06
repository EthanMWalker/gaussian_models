import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torchvision.transforms import transforms

import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from rnn_gmm import RnnGmm, NTCrossEntropyLoss

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

torch.autograd.set_detect_anomaly(True)

def get_mnist(batch_size):
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.30811,)),
      transforms.Lambda(lambda x: torch.flatten(x))
      # transforms.Lambda(lambda x: torch.unsqueeze(x,0))
    ]
  )

  train_set = tv.datasets.MNIST(
    'data/', train=True, download=True, transform=transform
  )
  test_set = tv.datasets.MNIST(
    'data/', train=False, download=True, transform=transform
  )

  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size
  )
  test_loader = torch.utils.data.DataLoader(
    test_set, batch_size
  )

  return train_loader, test_loader, train_set, test_set

def train(model, train_loader, lr, batch_size, num_epochs=10, save_iters=5):

  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  criterion = NTCrossEntropyLoss(.5, batch_size, device)
  losses = []
  running_loss = 0
  len_epoch = len(train_loader)

  with tqdm(total=num_epochs*len(train_loader)) as progress:
    for epoch in range(num_epochs):
      running_loss = 0.
      for i , (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        
        if x.size(0) != batch_size:
          continue

        mask = torch.round(torch.rand(x.size(1))).to(device)

        optimizer.zero_grad()
        sample = model.sample(x.size(0), x, mask)
        bool_mask = torch.gt(1-mask,0)
        compare = x[:,bool_mask]
        to_pad = x.size(1) - sample.size(1)
        sample = F.pad(sample, (1,to_pad), 'constant', 0)
        compare = F.pad(compare, (1,to_pad), 'constant', 0)
        loss = criterion(sample, compare)
        loss = loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        progress.set_description(
            f'l: {loss.item()} lr: {lr:.8} e: {epoch}'
          )
        progress.update()

        if i % 10 == 9:
          losses.append(running_loss / 100)
          running_loss = 0
          
      if epoch % save_iters == 0:
        torch.save(
          {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
          }, f'chkpt/backup/large_rnngmm_{lr:.8}_{epoch}.tar'
        )
      
  
  return model, losses


if __name__ == '__main__':
  batch_size = 1024

  train_loader, test_loader, train_set, test_set = get_mnist(batch_size)

  # for layers in [2**i for i in [2,3,4,5,6]]:
  for layers in [64]:
    for lr in [1e-5]:
      n_epochs = 100
      
      model = RnnGmm(
        28*28, 28, layers, 10, device
      ).to(device)
      
      model, losses = train(model, train_loader, lr, batch_size, n_epochs, 20)

      torch.save(
        {
        'model_state_dict': model.state_dict()
        }, f'chkpt/large_rnngmm_{lr:.8}.tar'
      )

      # model.load_state_dict(torch.load('chkpt/test.tar')['model_state_dict'])

      samples = model.sample(100).detach().cpu().numpy()
      out_dict = {
        'sample': samples,
        'losses': losses
      }

      with open(f'chkpt/large_rnngmm_{lr:.8}.pickle','wb') as out:
        pickle.dump(out_dict, out)
      
      try:
        plt.plot(losses)
        plt.savefig(f'chkpt/images/large_rnngmm_{lr:.8}.png')
      except:
        print('there was an exception')