import numpy as np
import torch 
import torch.nn.functional as F 
import torch.optim as optim
import util.utils as utils
import argparse 
from model.gcn import GCN
import time
from sklearn.model_selection import train_test_split

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--epochs', type=int, default=1000, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

adj, features, labels= utils.load_data()

index_train, index_test = train_test_split(range(labels.shape[0]), test_size=0.2, shuffle=False)
model = GCN(nfeat=features.shape[1], nhidden=args.hidden, nclass=labels.max().item() +1, dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if(args.cuda):
    print('Using cuda')
    model = model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
else:
    print('Using cpu')

print('parameter: ')
for i, p in enumerate(model.parameters()):
    print(i, p.size())
def train(features, adj, labels, index=range(features.shape[0])):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = F.nll_loss(output[index], labels[index])
    loss.backward()
    optimizer.step()
    return loss, time.time()- t

def test(features, adj, labels, index=range(features.shape[0])):
    model.eval()
    output = model(features, adj)
    loss = F.nll_loss(output[index], labels[index])
    acc = utils.accuracy(output[index], labels[index])
    return loss, acc

acc_best = 0
start = time.time()
for epoch in range(args.epochs):
    loss, t = train(features, adj, labels, index_train)
    loss_train, acc_train = test(features, adj, labels, index_train)
    loss_test, acc_test = test(features, adj, labels, index_test)
    if(acc_test>acc_best):
        acc_best = acc_test
    print('Epoch: [{}/{}], time: {:.3f}, loss_train: {:.3f}, accu_train: {:.3f}, loss_test: {:.3f}, acc_test: {:.3f}'.format(epoch, args.epochs, t, loss_train, acc_train, loss_test, acc_test))

print('Done!')
print('Total time: {:.3f}'.format(time.time()-start))
print('Best accuracy: {}'.format(acc_best))
