import argparse, configparser
import math
import threading
import torch
import socket
import datetime
import os
import re

from transformer import Transformer
from dataset import LTMBDataset
from trainer import Trainer

def parse_int_pair(pair_str):
    if pair_str.lower() == 'all':
        return 'all'
    
    pattern = re.compile(r'\((\d+),(\d+)\)')
    match = pattern.match(pair_str)
    
    if match:
        x, y = map(int, match.groups())
        return x, y
    else:
        raise argparse.ArgumentTypeError("Invalid integer pair format. Must be in the form '(x,y)'.")


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='[Optional] Path to config file. Note that config file arguments override command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--num_heads', type=int, default=2, help='number of heads for multi-head attention')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers for transformer')
parser.add_argument('--memory_multiplier', type=float, default=10.0, help='multiplier for memory loss')
parser.add_argument('--ckpt_folder', type=str, default=None, help='folder to save checkpoints and logs')
parser.add_argument('--viz_every', type=int, default=100, help='visualize attention every n iterations')
parser.add_argument('--ckpt_every', type=int, default=100, help='save checkpoint every n epochs')
parser.add_argument('--eval_every', type=int, default=5, help='save checkpoint every n epochs')
parser.add_argument('--load_ckpt', type=str, default=None, help='checkpoint to load from')
parser.add_argument('--benchmark', type=str, choices=['LTMB-Hallway-v0', 'LTMB-Ordering-v0', 'LTMB-Couting-v0'], help='benchmark to evaluate on')
parser.add_argument('--dataset', type=str, help='path to dataset of expert demonstrations')
parser.add_argument('--test_dataset', type=str, help='path to test dataset of expert demonstrations')
parser.add_argument('--length', type=int, default=None, help='max length of episode')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')
parser.add_argument('--memory_loss_heads', metavar='(layer,head)', type=parse_int_pair, nargs='+', default='all', help='Specify heads to apply the memory loss on. Input integer pairs in the form (x,y) or "all" to specify all heads. Layers and heads are 1-indexed.')
parser.add_argument('--viz_heads', metavar='(layer,head)', type=parse_int_pair, nargs='+', default='all', help='Specify heads to visualize attention heatmap. Input integer pairs in the form (x,y) or "all" to specify all heads. Layers and heads are 1-indexed.')
args = parser.parse_args()

if args.config is not None:
    config = configparser.ConfigParser()
    config.read(args.config)
    defaults = dict(config.items("Defaults"))
    if 'memory_loss_heads' in defaults and defaults['memory_loss_heads'] != 'all':
        defaults['memory_loss_heads'] = [parse_int_pair(pair) for pair in defaults['memory_loss_heads'].split()]
    if 'viz_heads' in defaults and defaults['viz_heads'] != 'all':
        defaults['viz_heads'] = [parse_int_pair(pair) for pair in defaults['viz_heads'].split()]
    parser.set_defaults(**defaults)
    args = parser.parse_args()
    
# check for required arguments
if any(arg is None for arg in [args.benchmark, args.dataset, args.test_dataset, args.length]):
    raise ValueError('Missing one of the required arguments: --benchmark, --dataset, --test_dataset, --length')

# process user specified attention heads
def process_heads(heads):
    if heads == 'all':
        heads = [(layer, head) for layer in range(1, args.num_layers + 1) for head in range(1, args.num_heads + 1)]
    for layer, head in heads:
        if layer < 1 or layer > args.num_layers:
            raise ValueError(f'Invalid layer {layer}. Must be between 1 and {args.num_layers}.')
        if head < 1 or head > args.num_heads:
            raise ValueError(f'Invalid head {head}. Must be between 1 and {args.num_heads}.')
    return heads

args.memory_loss_heads = process_heads(args.memory_loss_heads)
args.viz_heads = process_heads(args.viz_heads)

# create runs folder
if not os.path.exists('runs'):
    os.mkdir('runs')

# create ckpts folder
if args.ckpt_folder is None:
    args.ckpt_folder = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + f'-{socket.gethostname()}'
args.ckpt_folder = 'runs/' + args.ckpt_folder

torch.manual_seed(args.seed)

# compute max length of episode and action dimension
max_len = None
if args.benchmark == 'LTMB-Hallway-v0':
    max_len = 4 * args.length + 25
elif args.benchmark == 'LTMB-Ordering-v0':
    max_len = 18 + args.length
elif args.benchmark == 'LTMB-Counting-v0':
    max_len = 7 * args.length
assert max_len is not None
action_dim = 7

train_dataset = LTMBDataset(args.dataset, max_len)
test_dataset = LTMBDataset(args.test_dataset, max_len)
model = Transformer(action_dim, args.d_model, args.num_heads, args.num_layers, max_len).cuda()
trainer = Trainer(model, train_dataset, test_dataset, args)
if args.load_ckpt is not None:
    trainer.load(args.load_ckpt)

# train
trainer.train(args.epochs)
trainer.save('final')