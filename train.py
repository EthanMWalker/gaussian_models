import torch
import torchvision as tv
from torchvision.transforms import transforms

import pickle
from tqdm import tqdm

from gmm import RnnGmm

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



def get_mnist(batch_size):
  transform = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.30811,))
    ]
  )

  train_set = tv.datasets.MNIST(
    '../data/', train=True, download=True, transform=transform
  )
  test_set = tv.datasets.MNIST(
    '../data/', train=False, download=True, transform=transform
  )

  train_loader = torch.utils.data.DataLoader(
    train_set, batch_size
  )
  test_loader = torch.utils.data.DataLoader(
    test_set, batch_size
  )

  return train_loader, test_loader, train_set, test_set

def train(model, train_loader, lr, num_epochs=10, save_iters=5):

  optimizer = torch.optim.SGD(model.parameters(), lr=lr)
  losses = []
  running_loss = 0

  with tqdm(total=num_epochs) as progress:
    for epoch in range(num_epochs):
      for i , (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        loss = model.log_prob(x)
        loss = -loss.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 99:
          losses.append(running_loss / 100)
          running_loss = 0
          
      for i , (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        z, log_det = model(x)
        model.update_gmm(z)

      progress.set_description(
          f'l: {model.num_layers} lr: {lr:.8} e: {epoch}'
        )
      progress.update()

      if epoch % save_iters == 0:
        torch.save(
          {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss
          }, f'chkpt/backup/mnist_gmm_{model.num_layers}_{lr:.8}_{epoch}.tar'
        )
      
  
  return model, losses


if __name__ == '__main__':
  batch_size = 512
  res_net_layers = 10

  train_loader, test_loader, train_set, test_set = get_mnist(batch_size)

  # for layers in [2**i for i in [2,3,4,5,6]]:
  for layers in [1,2,4,8,16, 32]:
    for lr in [1e-10, 1e-8]:
      n_epochs = 100
      
      model = RealNVP(
        1, 5, layers, 10, (1,28,28), device, res_net_layers
      ).to(device)

      model, losses = train(model, train_loader, lr, n_epochs, 100)

      torch.save(
        {
        'model_state_dict': model.state_dict()
        }, f'chkpt/mnist_gmm_{layers}_{lr:.8}.tar'
      )

      # model.load_state_dict(torch.load('chkpt/test.tar')['model_state_dict'])

      samples = model.sample(100).detach().cpu().numpy()
      out_dict = {
        'sample': samples,
        'losses': losses
      }

      with open(f'chkpt/mnist_gmm_samples_{layers}_{lr:.8}.pickle','wb') as out:
        pickle.dump(out_dict, out)