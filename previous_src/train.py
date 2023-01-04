import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    elif isinstance(x, (list, tuple)):
        x = [to_device(i, device) for i in x]
    elif isinstance(x, dict):
        for k,v in x.items():
            x[k] = to_device(v, device)
    return x 

def train(Dataset, Model, epoch=10000, eval_every_eps=1, save_every_eps=1, lr=1e-3, weight_decay=5e-4, device='cpu'):
    model = Model(
        input_size=Dataset.size,
        n_classes =Dataset.n_classes, 
        n_channels=Dataset.n_channels,
        n_branch  =Dataset.n_branch 
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    metric_fn = lambda p, y: f1_score(y.cpu().numpy(), p.argmax(-1).cpu().numpy(), average='micro')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = Dataset.loader('train')
    valid_loader = Dataset.loader('valid')

    for ep in range(epoch):

        for feats, label in tqdm(train_loader, colour='blue', desc=f'[Epoch{ep:^3}] Train'):
            feats, label = to_device([feats,label], device)
            model.train()
            logits = model(feats)
            loss   = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        if (ep + 1) % eval_every_eps == 0:
            l_metric = np.array([])
            for feats, label in tqdm(valid_loader, colour='green', desc=f'[Epoch{ep:^3}] Valid'):
                feats, label = to_device([feats,label], device)
                model.eval()
                with torch.no_grad():
                    logits = model(feats)
                    metric = metric_fn(logits, label)
                l_metric = np.append(l_metric, metric)
            print(f"metric:{l_metric.mean()}({l_metric.std()})")

        # if (ep + 1) % save_every_eps == 0:
        #     model.save(affix=f"[Epoch:{ep}]")

        if (ep + 1) % 2000 == 0:
             model.save(affix=f"[Epoch:{ep}]")

    model.save(affix="Finish")

if __name__ == '__main__':
    from dataset import MyKnees3D
    from model import BranchViT

    train(MyKnees3D, BranchViT, device='cuda:0')