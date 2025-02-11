#import pymatgen.core.periodic_table
import torch
import torch.nn as nn
import sys
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
#import wandb
import argparse 
import os
import time
import json
import GPUtil
import lib.models as libmod
import lib.tables
import sklearn.metrics



device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
#device = "cpu"
timer = 0
maxload = 0.7
maxmem = 0.7
while(not GPUtil.getAvailable("load", maxLoad = maxload, maxMemory = maxmem)):
    print("\rWaiting for GPU, {} seconds".format(timer), end = "", flush = True)
    time.sleep(1)
    timer += 1
torch.cuda.set_device(GPUtil.getAvailable("load", maxLoad = maxload, maxMemory = maxmem)[0])

def main():
    torch.manual_seed(123)
    random.seed(567)
    
    ##wandb stuff
    #wandb.init()
    argv = sys.argv #wandb.config.argv
    print(argv)
    #batch_size = 1000 #wandb.config.batch_size
    #lr = 1e-4 #wandb.config.lr
    #wd = 1e-6 #wandb.config.wd
    #droprate = 0.1 #wandb.config.dropout
    #nn_width = 2048 #wandb.config.nn_width
    #epochs = 50 #wandb.config.epochs
    #funnel = 8 #wandb.config.funnel
    #tvsplit = 0.8
    #ams = True #wandb.config.ams
    #log = False
    
    #parse args
    parser = argparse.ArgumentParser(description='AB x Linear')
    parser.add_argument('root_dir', metavar='OPTIONS', help='path to root dir, '
                         'then other options')
    parser.add_argument('--id-prop-p', metavar='N', default='id_clean_v.csv', type=str, help="Cif list (csv) prediction")
    parser.add_argument("--out", metavar = "N", default="outputs", type = str, help = "output folder location")
    parser.add_argument("-d", metavar = "N", default = 0.00, type = float, help = "dropout")
    parser.add_argument("--width", metavar = "N", default = 699, type = int, help = "netwidth")
    parser.add_argument("--funnel", metavar = "N", default = 2, type = int, help = "funnel rate")
    parser.add_argument("-b", metavar = "N", default = 1000, type = int, help = "batch size")
    parser.add_argument("-m", metavar = "N", default = 0, type = int, help = "model type: 0: linear, 1: +pooling")
    parser.add_argument("-c", metavar = "N", default = 3, type = int, help = "conv layer count")
    parser.add_argument("--ari", metavar = "N", default = "base", type = str, help = "ari mod")
    parser.add_argument("--bondlen", metavar = "N", default = 0, type = float, help = "shrink globals by")
    parser.add_argument("--log", action = "store_true", help = "log the targets")
    args, unknowns = parser.parse_known_args(sys.argv[1:])
    
    droprate = args.d
    nn_width = args.width
    funnel = args.funnel
    batch_size = args.b
    nconv = args.c
    bondlenchk = args.bondlen
    log = args.log
    logoffset = 5
    
    prefix = args.root_dir
    val_file = prefix + "/" + args.id_prop_p
    #load atominit
    print("Using {} device {}".format(device, GPUtil.getAvailable("load")[0]))
    outfold = "./train_outputs/{}".format(args.out)
    model_file_v = outfold + "/" + "model_min_V.pth"
    val_pred_ofile = outfold + "/" + "model_predictions.csv"
    normfile = outfold + "/" + "normalizer.json"
    
    if not os.path.exists(outfold):
        os.makedirs(outfold)
    
    ari = json.load(open('atom_init_{}.json'.format(args.ari), 'r'))
    arilen = len(ari["1"])
    
    database_v = []
    with open(val_file, "r") as idprop:
        for line in idprop:
            buffer = line.strip().split(",")
            atoms = buffer[2:4]
            if bondlenchk == 0:
                bondlen = buffer[4:]
            elif bondlenchk == 1:
                bondlen = buffer[4:-1]
            elif bondlenchk == 2:
                bondlen = buffer[4:-2]
            features = []
            for i in atoms:
                features.extend(ari[str(lib.tables.periodic_table[i])])
            #database[buffer[0]] = np.array(features)
            #id_prop[buffer[0]] = buffer[1]
            features.extend(bondlen)
            target = np.float32([buffer[1]])
            if log:
                target = np.log(target + logoffset)
            database_v.append([np.float32(features), target, buffer[0]])
    
    try:
        norm_dict = json.load(open(normfile, 'r'))
        normalizer = Normalizer([1,2,3])
        normalizer.load_state_dict(norm_dict)
        
        for i in range(len(database_v)):
            database_v[i][1] = normalizer.norm(database_v[i][1])
        normalize = True
    except:
        normalize = False
    
    if args.m == 0:
        model = libmod.SimpleNetwork(len(database_v[0][0]), droprate, nn_width, funnel).to(device)
    elif args.m == 1:
        model = libmod.ConvNetwork(len(database_v[0][0]), droprate, nn_width, funnel, arilen).to(device)
    elif args.m == 2:
        model = libmod.PoolConvNetwork(len(database_v[0][0]), droprate, nn_width, funnel, arilen).to(device)
        #model.init_weights(nn.init.normal_,0,2)
    elif args.m == 3:
        model = libmod.BilinNetwork(len(database_v[0][0]), droprate, nn_width, funnel, arilen).to(device)
    elif args.m == 4:
        model = libmod.SimpleGNN(len(database_v[0][0]), droprate, nn_width, funnel, arilen, nconv).to(device)
    elif args.m == 5:
        model = libmod.AtomPoolGNN(len(database_v[0][0]), droprate, nn_width, funnel, arilen, nconv).to(device)
    elif args.m == 6:
        model = libmod.AtomPoolGNN2(len(database_v[0][0]), droprate, nn_width, funnel, arilen, nconv).to(device)
    elif args.m == 7:
        model = libmod.PoolNetwork(len(database_v[0][0]), droprate, nn_width, funnel, arilen).to(device)
    elif args.m == 8:
        model = libmod.LinearModel(len(database_v[0][0])).to(device)
    elif args.m == 9:
        model = libmod.AtomPoolGNN_numsum(len(database_v[0][0]), droprate, nn_width, funnel, arilen, nconv).to(device)
    else:
        print("model type not found")
        return 1
    #print(model)
    
    #targtetlist = [i[1] for i in (database_t + database_v).random.sample(len(database_t))]
    #normalizer = Normalizer(targetlist)
    
    valloader = DataLoader(database_v, batch_size = batch_size, shuffle = True)
    
    loss_T = []
    loss_V = []
    mae_T = []
    mae_V = []
    tstart = time.time()
    
    model.load_state_dict(torch.load(model_file_v))
    model.eval()
          
    with open(val_pred_ofile, "w") as outfile:
        for batch, (in_fea, target, ids) in enumerate(valloader):
            #print(in_fea)
            x = in_fea.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            if normalize:
                if log:
                    for i in zip(ids, normalizer.denorm((np.exp(target[:,0].detach()) - logoffset).tolist()), normalizer.denorm((np.exp(pred[:,0].detach().tolist()) - logoffset))):
                        outfile.write(",".join(map(str, i))+"\n")
                else:
                    for i in zip(ids, normalizer.denorm(target[:,0].detach().tolist()), normalizer.denorm(pred[:,0].detach().tolist())):
                        outfile.write(",".join(map(str, i))+"\n")
            else:
                if log:
                    for i in zip(ids, (np.exp(target[:,0].detach()) - logoffset).tolist(), (np.exp(pred[:,0].detach().tolist()) - logoffset)):
                        outfile.write(",".join(map(str, i))+"\n")
                else:
                    for i in zip(ids, target[:,0].detach().tolist(), pred[:,0].detach().tolist()):
                        outfile.write(",".join(map(str, i))+"\n")


def train(dataloader, model, lossfn, optimizer):
    nbatch = len(dataloader)
    model.train()
    for batch, (in_fea, target, ids) in enumerate(dataloader):
        #print(in_fea)
        x = in_fea.to(device)
        y = target.to(device)
        #print("{}, {}, {}".format(batch, target, ids))
        pred = model(x)
        #print(pred)
        loss = lossfn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4)
        optimizer.step()
    return loss, sklearn.metrics.mean_absolute_error(target, pred.cpu().detach().numpy())
    
def test(dataloader, model, lossfn):
    nbatch = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch, (in_fea, target, ids) in enumerate(dataloader):
            #print(target)
            x = in_fea.to(device)
            y = target.to(device)
            #print("{}, {}, {}".format(batch, target, ids))
            pred = model(x)
            test_loss += lossfn(pred, y).item()
    return test_loss/nbatch, sklearn.metrics.mean_absolute_error(target, pred.cpu().detach().numpy())
    

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = np.mean(tensor, axis = 0)
        self.std = np.std(tensor, axis = 0)
        self.std[self.std == 0] = 1

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean.tolist(),
                'std': self.std.tolist()}

    def load_state_dict(self, state_dict):
        self.mean = np.array(state_dict['mean'])
        self.std = np.array(state_dict['std'])

# sweep_configuration = {
    # 'method': 'bayes',
    # 'name': 'Bayesian Search 5 val',
    # 'metric': {'goal': 'minimize', 'name': 'val_loss'},
    # 'parameters': 
    # {
        # 'batch_size': {'values': [5, 10, 20, 40, 80]},
        # #'epochs': {'values': [1000, 2000, 5000]},
        # 'lr': {'max': 0.001, 'min': 0.000001},
        # 'wd': {'max': 0.00001, 'min': 0.000001},
        # 'dropout': {'max': 0.1, 'min': 0.0},
        # #'nn_width': {'values': [128, 256, 512, 1024]},
        # #"funnel": {'values': [8]},
        # "argv":{"values": [sys.argv]},
        # "ams":{"values": [True, False]}
    # },
    # "program": "ABlinear_nn.py",
# }

# sweep_id = wandb.sweep(
  # sweep=sweep_configuration, 
  # project='AB Linear Networks'
  # )

#wandb.agent(sweep_id, function = main)
main()
