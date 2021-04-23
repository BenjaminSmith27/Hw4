import pandas as pd
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import DistributedOptimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from time import perf_counter
#model class
class SmithNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1,4,3)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(4,10,3)
        self.pool = nn.AdaptiveMaxPool2d((1,1))
        self.classifier = nn.Linear(10, num_classes)
    def forward(self,x):
        x=self.conv1(x)
        x=self.activation(x)
        x=self.conv2(x)
        x=self.activation(x)
        x=self.pool(x)
        x=x.squeeze(-1).squeeze(-1)
        x=self.classifier(x)
        return x
    
#main training loop
def train(num_epochs=30, batch_size=128):
    #set up MINIST
    train_set = MNIST(root ='.', train=True, transform=transforms.Compose([transforms.ToTensor(),]) ,download=True)
    test_set = MNIST(root='.', train=False, download=True)
    #dataloader
    sampler = DistributedSampler(train_set)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    #instantiate model from class
    model=SmithNet(num_classes=10)
    model=DistributedDataParallel(model)
    #set up optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    #set up loss
    criterion = nn.CrossEntropyLoss()
    rank = dist.get_rank()
    #loop for epochs
    iter_losses = []
    epoch_losses = []
    elapsed_times=[]
    epbar = range(num_epochs)
    start = perf_counter()
    for ep in epbar:
        #loop over data (minibatch)
        ep_loss = 0.0
        num_iters= 0
        for X, Y in train_dataloader:
            #zero gradients
            optimizer.zero_grad()
            #eval model
            pred = model(X)
            #compare w labels
            loss = criterion(pred,Y)
            #print loss
            iter_losses.append(loss.item())
            ep_loss += loss.item()
            #compute gradients
            loss.backward()
            #step optimizer
            optimizer.step()
            num_iters += 1
        ep_loss /= num_iters
        #print epoch loss
        if rank ==0:
            print("epoch", ep, "num_iters", num_iters, "loss", ep_loss, "elapsed time (s)", perf_counter() - start)
            
        elapsed_times.append(perf_counter() - start)
        epoch_losses.append(ep_loss)
        metrics = pd.DataFrame({'epoch_losses': epoch_losses, 'elapsed_time': elapsed_times})
        metrics.to_csv('metrics8.csv')

    return iter_losses, epoch_losses

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    import mpi4py.MPI as MPI
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    
    if rank ==0:
        print("world size:", world_size)
    dist.init_process_group('gloo', init_method='env://', world_size=world_size, rank=rank,)
    
    train()
