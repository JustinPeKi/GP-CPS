
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import random
import glob
import argparse
from solver import Solver
import os
from data_loader.kvasir_seg import Kvasir_SEG_dataset
from model.unet import MyUNet

torch.cuda.set_device(0)  # GPU id
torch.cuda.empty_cache()
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description='Latent Interaction Supervision')

    parser.add_argument('--tip', type=str, default='')

    # dataset info
    parser.add_argument('--dataset', type=str, default='Kvasir-SEG',
                        help='retouch-Spectrailis,retouch-Cirrus,retouch-Topcon, isic, chase')
    parser.add_argument('--data_root', type=str, default='data_loader/Kvasir-SEG',
                        help='dataset directory')
    parser.add_argument('--resize', type=int, default=[256, 256], nargs='+',
                        help='image size: [height, width]')

    # network option & hyper-parameters
    parser.add_argument('--num-class', type=int, default=1, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--trainset-size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-update', type=str, default='step',
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts')
    parser.add_argument("--cuda-id", type=int, default=0)
    parser.add_argument('--lam-sup', type=float, default=1)
    parser.add_argument('--lam-semi', type=float, default=0)
    parser.add_argument('--lam-latent', type=float, default=0)
    parser.add_argument('--lam-dice', type=float, default=0)
    parser.add_argument('--lam-ce', type=float, default=0)
    parser.add_argument('--lam-lip', type=float, default=0)
    parser.add_argument('--lam-mix', type=float, default=0)
    parser.add_argument('--grad-edge', type=float, default=10)
    parser.add_argument('--grad-inside', type=float, default=10)
    parser.add_argument('--grad-outside', type=float, default=0)

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--weights', type=str,
                        default='retouch_weights',
                        help='path of SDL weights')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--exp-id', type=int, default=100)
    parser.add_argument('--save-per-epochs', type=int, default=10000,
                        help='per epochs to save')

    # evaluation only
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='test only, please load the pretrained model')

    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    return args


def main(args):
    set_seed(727)
    torch.cuda.set_device(args.cuda_id)
    if args.dataset == 'Kvasir-SEG':
        trainset_labeled = Kvasir_SEG_dataset(folder=args.data_root, start=0, end=args.trainset_size)
        trainset_unlabeled = Kvasir_SEG_dataset(folder=args.data_root, start=args.trainset_size, end=800)
        validset = Kvasir_SEG_dataset(folder=args.data_root, start=800, end=1000)

    else:
        trainset_labeled = Kvasir_SEG_dataset(folder=args.data_root, start=0, end=199)
        trainset_unlabeled = Kvasir_SEG_dataset(folder=args.data_root, start=200, end=799)
        validset = Kvasir_SEG_dataset(folder=args.data_root, start=800, end=999)

    train_labeled_loader = (torch.utils.data.DataLoader(dataset=trainset_labeled,
                                                        batch_size=int(args.batch_size*len(trainset_labeled)/(len(trainset_labeled)+len(trainset_unlabeled))),
                                                        shuffle=True))
    train_unlabeled_loader = torch.utils.data.DataLoader(dataset=trainset_unlabeled,
                                                         batch_size=int(args.batch_size*len(trainset_unlabeled)/(len(trainset_labeled)+len(trainset_unlabeled))),
                                                         shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False)

    print("Train labeled batch number: %i" % len(train_labeled_loader))
    print("Train unlabeled batch number: %i" % len(train_unlabeled_loader))
    print("Test batch number: %i" % len(val_loader))

    #### Above: define how you get the data on your own dataset ######
    set_seed(727)
    model1 = MyUNet(in_channels=3, out_channels=args.num_class).cuda()
    set_seed(506)
    model2 = MyUNet(in_channels=3, out_channels=args.num_class).cuda()

    if args.pretrained:
        pass
        # model.load_state_dict(torch.load(args.pretrained, map_location=torch.device('cpu')))
        # model = model.cuda()

    solver = Solver(args)
    set_seed(727)
    # set_seed(2048)
    solver.train(model1, model2, train_labeled_loader, train_unlabeled_loader, val_loader, num_epochs=args.epochs,
                 exp_id=args.exp_id)


if __name__ == '__main__':
    args = parse_args()
    main(args)
