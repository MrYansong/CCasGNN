#encoding:utf-8

import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description="Run CCasGNN.")

    parser.add_argument('--graph-file-path',
                        nargs='?',
                        default='./weibo/weibo_24hours_obs120.json',
                        help='Folder with graph pair jsons.')
    parser.add_argument('--result-log',
                        type=str,
                        default='./log/weibo_24hours_obs120.log',
                        help='')
    parser.add_argument('--number-of-features',
                        type=int,
                        default=2,
                        help='')
    parser.add_argument('--number-of-nodes',
                        type=int,
                        default=100,
                        help='')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help="Number of training epochs. Default is 100.")
    parser.add_argument('--check-point',
                        type=int,
                        default=5,
                        help="")
    parser.add_argument('--train-ratio',
                        type=float,
                        default=0.7,
                        help='')
    parser.add_argument('--valid-ratio',
                        type=float,
                        default=0.1,
                        help='')
    parser.add_argument('--test-ratio',
                        type=float,
                        default=0.2,
                        help='')
    parser.add_argument('--batch-size',
                        type=int,
                        default=100,
                        help='')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.005,
                        help='')
    parser.add_argument('--user-embedding-dim',
                        type=int,
                        default=8,
                        help='the number of features')
    parser.add_argument('--location-embedding-dim',
                        type=int,
                        default=16,
                        help='')
    parser.add_argument('--gcn-out-channel',
                        type=int,
                        default=32,
                        help="gcn out nodes feature size")
    parser.add_argument('--gat-n-heads',
                        type=int,
                        default=2,
                        help="")
    parser.add_argument('--gcn-filters-1',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--gcn-filters-2',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.1,
                        help="")
    parser.add_argument('--att-num-heads',
                        type=int,
                        default=2,
                        help="")
    parser.add_argument('--att-dim-k',
                        type=int,
                        default=16,
                        help="")
    parser.add_argument('--att-dim-v',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--attn-dropout',
                        type=float,
                        default=0.2,
                        help="")
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.001,
                        help="Adam weight decay. Default is 0.001.")
    parser.add_argument('--dens-hiddensize',
                        type=int,
                        default=32,
                        help="")
    parser.add_argument('--dens-dropout',
                        type=float,
                        default=0.1,
                        help="")
    parser.add_argument('--dens-outsize',
                        type=int,
                        default=1,
                        help="")

    return parser.parse_args()