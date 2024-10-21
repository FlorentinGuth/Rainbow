import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
import argparse
import sys

import utils
utils.do_svd = False
import notebook.experiments
notebook.experiments.tqdm_enabled = False
from notebook.experiments import *
from notebook.plots import *


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", help="name of the experiment to resample")
parser.add_argument("--dir", help="name of log file")
parser.add_argument("--epoch", default=None, type=int, help="epoch of model to use (-1 for last)")
parser.add_argument("--num-atoms", default=None, type=int, nargs="+", help="change number of atoms")
parser.add_argument("--align-test", action="store_true", help="compute alignment on the test set")
parser.add_argument("--num-simultaneous", type=int, default=1, help="number of simultaneous resamplings")
parser.add_argument("--num-repeats", type=int, default=1, help="number of total resamplings")
parser.add_argument("--data", default=None, help="optional path to imagenet")
parser.add_argument("--batch-size", default=128, type=int, help="batch size for computing alignment")
args = parser.parse_args()


# Little snippet to pipe stdout and stderr to a file. Does not capture warning (should probably use the logging module, but more complicated...).

class Logger(object):
    def __init__(self, terminal):
        self.terminal = terminal
        self.log = open(args.dir, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(sys.stdout)
sys.stderr = Logger(sys.stderr)


if args.data is not None:
    exps.imagenet_path = args.data
if args.epoch == -1:
    args.epoch = None
print(args)

aligned_exps = {args.experiment:
                AlignedExperiments(exps.load_pattern(
                    args.experiment, log=True, num=1, full_names=True,
                    ), num_iters=-1)}

num_atoms = None if args.num_atoms is None else {j + 1: args.num_atoms[j] for j in range(len(args.num_atoms))}
methods = ["gaussian"]
resamples = {f"{pattern}-{method}-try{i}": (pattern, method, num_atoms) 
        for pattern in aligned_exps.keys() for method in methods for i in range(args.num_simultaneous)}
print(resamples)

for i in range(args.num_repeats):
    new_models, checkpoint_dicts, results = resample(aligned_exps, resamples=resamples, batch_size=args.batch_size, 
            align_test=args.align_test, epoch=args.epoch)
    torch.save(results, f"{args.dir}_results_repeat{i}.pt")
    # This overrides to keep only the last checkpoint dict (which should suffice for our practices).
    for key, checkpoint_dict in checkpoint_dicts.items():
        torch.save(checkpoint_dict, f"{args.dir}_checkpoint_repeat{i}.pt")

