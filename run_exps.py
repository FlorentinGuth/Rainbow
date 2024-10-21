import numpy as np
import shlex
import subprocess
import argparse


# Specify the path to the ImageNet dataset here.
imagenet_path = "PATH/TO/IMAGENET"


parser = argparse.ArgumentParser()
parser.add_argument("--print", action="store_true", help="only print the commands instead of running them")
args = parser.parse_args()


# Utilities to generate experiment commands.


class Exps:
    def __init__(self, exps):
        self.exps = exps  # List of exps, each exp being represented as tuple(names, cmds) which represent name and cmd parts to join.

    def __or__(self, other):
        return Exps(self.exps + other.exps)

    def __mul__(self, other):
        return Exps([(name1 + name2, cmd1 + cmd2) for name1, cmd1 in self.exps for name2, cmd2 in other.exps])

    def run(self):
        for i, (names, cmds) in enumerate(self.exps, start=1):
            name = "-".join(filter(None, names))
            cmd = " ".join(filter(None, cmds))
            print(f"[{i:{len(str(len(self.exps)))}}/{len(self.exps)}] {name}:\t{cmd}")
            if not args.print:
                cmd = f"{cmd} --dir {name}"
                print(str(subprocess.check_output(shlex.split(cmd))))

    @property
    def name(self):
        """ If single exp, returns its name. """
        assert len(self.exps) == 1
        return "-".join(filter(None, self.exps[0][0]))


def exp(name="", cmd=""):
    return Exps([([name], [cmd])])


single_empty_exp = exp()  # A single exp with no arguments nor name
empty_exps = Exps([])  # An empty list of experiments
exps = empty_exps


def ors(exps):
    res = empty_exps
    for exp in exps:
        res = res | exp
    return res


def base(dataset, grayscale=True, standard=False):
    """ Creates a base experiment with default training arguments depending on the dataset. """
    num_jobs = dict(cifar=2, cifar100=2, mnist=2, imagenet=10, imagenet32=2)[dataset]
    dataset_args = dict(cifar="--cifar10", cifar100="--cifar100", mnist="--mnist", imagenet=f"--data {imagenet_path}", imagenet32="--dataset ImageNet32")[dataset]

    if standard:
        args = f" -p 1000 --epochs 90 --learning-rate-adjust-frequency 30 --lr 0.1 -j {num_jobs} --batch-size 256 {dataset_args}"
    else:
        args = f" -p 1000 --epochs 150 --learning-rate-adjust-frequency 50 --lr 0.01 -j {num_jobs} --batch-size 128 {dataset_args}"

    base_exp = exp(cmd=f"python main_block.py{args}", name=dataset if not standard else "standard")

    if not standard and grayscale and dataset != "mnist":
        base_exp *= exp(cmd="--grayscale", name="grayscale")

    return base_exp


def blocks(num_blocks, L):
    """ L can either be an integer or an iterable """
    if isinstance(L, int):
        name = f"L{L}"
        L = [L] * num_blocks
    else:
        assert len(L) == num_blocks
        name = f"L{'L'.join(str(l) for l in L)}"
    return exp(cmd=f"--n-blocks {num_blocks} --scattering-wph {' '.join(['1'] * num_blocks)} --scat-angles {' '.join([str(l) for l in L])}", name=name)


def arch(projs, L, skip=False, double_scales=False, factorize_filters=None, complex=False, std=True, norm=False):
    P = "Pc" if complex else "Pr"
    res = single_empty_exp
    res *= exp(name="".join("W" + (f"P{'c' if complex else ''}{p}" if p != "id" else "") for p in projs))
    if skip:
        res *= exp(name=f"skip{'' if std else '-nostd'}{'-norm' if norm else ''}", cmd=f"-a 'Fw rho {'Std ' if std else ''}{P}{' N' if norm else ''}'")
    else:
        res *= exp(cmd=f"-a 'Fw {'Std ' if std else ''}{P}{' N' if norm else ''}' --psi-arch 'mod'", name=f"{'' if std else 'nostd'}{'-' if norm and not std else ''}{'norm' if norm else ''}")
    res *= exp(cmd=f"--{P}-size {' '.join([str(p) for p in projs])}")
    if double_scales:
        if factorize_filters is None:
            factorize_filters = len(projs) % 2
        res *= exp(cmd=f"--factorize-filters {factorize_filters}")
    return res * blocks(len(projs), L)


cifar_sizes = [64, 128, 256, 512, 512, 512, 512, 512]
imagenet_sizes = [32, 64, 64, 128, 256, 512, 512, 512, 512, 512, 256]


def change_size(sizes, js, s):
    """ Multiplies all sizes by s at the given js. """
    return [int(size * s) if j in js else size for j, size in enumerate(sizes, start=1)]


def iclr(dataset, skip=False, complex_p=False, norm=False, std=True, L=4,
        dataset_train=None, width_scaling=1):
    """ Creates an ICLR-like experiment. """
    if dataset_train is None:
        dataset_train = dataset
    exp = base(dataset_train, grayscale=False)
    projs = dict(cifar=cifar_sizes, imagenet=imagenet_sizes)[dataset]
    projs = [int(width_scaling * p) for p in projs]
    exp *= arch(projs=projs, L=L, double_scales=True, complex=complex_p, std=std, norm=norm, skip=skip)
    # Not implemented: default angles (8 for ImageNet, 16 + full_angles for CIFAR)
    # Not implemented: training scheme: 150/45 for ImageNet, 300/70 for CIFAR
    # Assumes default lr, momentum, weight_decay, batch_size
    # Does not set name
    return exp


no_weight_decay = exp(cmd="--weight-decay 0", name="nowd")
inits = lambda n, prev=0: ors([exp(name=f"init{i+1}") for i in range(prev, n)])
classifier_bn = lambda bn: exp(name=f"clbn{bn}", cmd=f"--classifier-batch-norm {bn}")
classifier_nobias = exp(name=f"clnobias", cmd="--no-classifier-bias")

resnet = lambda depth=18: base("imagenet", grayscale=False, standard=True) * exp(name=f"resnet{depth}", cmd=f"--arch resnet{depth}-custom")
no_biases = exp(name="nobias-clnobias", cmd="--standard-no-bias --standard-classifier-no-bias")
batch_norm = lambda bn: exp(name=f"bn{bn}", cmd=f"--standard-batch-norm {bn}")


# EXPERIMENTS


# Gaussianity test : training 50 CIFAR J8 experiments
exps |= iclr("cifar", norm=True) * classifier_bn("std") * classifier_nobias * exp(name="gaussiantest") * inits(50)


# Resampling: first train base models.
exps |= ors(iclr("cifar", norm=True, skip=True, width_scaling=2 ** i) for i in range(-3, 4)) * classifier_bn("std") * classifier_nobias  # * exp(cmd="--batch-size 64")  # Uncomment if memory is an issue
# Then resample them.
base_rsmpl_cifar = exp(name="resamplings/resample", cmd="python resample.py")
rsmpl_exp = lambda name: exp(name=name, cmd=f"--experiment {name}")
rsmpl_cifar8 = lambda w, skip=False: rsmpl_exp((iclr("cifar", norm=True, width_scaling=w, skip=skip) * classifier_bn("std") * classifier_nobias).name)
exps |= base_rsmpl_cifar * ors(rsmpl_cifar8(2 ** i, skip=True) for i in range(-3, 4))  # * exp(cmd="--batch-size 32")  # Uncomment if memory is an issue
# Then optionally retrain the classifier.
retrain_class = lambda path, J, name: exp(cmd=f"--freeze-P {' '.join(['1'] * (J - 1))} --resume {path} --restart", name=f"load-{name}-retrainclf")
exps |= ors(iclr("cifar", norm=True, skip=True, width_scaling=2 ** i) * classifier_bn("std") * classifier_nobias * retrain_class(path=(base_rsmpl_cifar * rsmpl_cifar8(2 ** i, skip=True)).name + "_checkpoint_repeat0.pt", J=8, name="gaussian") for i in range(-3, 4))


# Activation convergence: training scattering networks on CIFAR and ResNet on ImageNet at various width scalings.
exps |= ors(iclr("cifar", norm=True, width_scaling=2 ** i) for i in range(-3, 4)) * classifier_bn("std") * classifier_nobias
width_scaling = lambda s: exp(name=f"widthX{s}", cmd=f"--standard-width-scaling {s}")
exps |= resnet(18) * no_biases * batch_norm("post") * ors(width_scaling(2 ** i) for i in range(-3, 4)) * exp(name="fix")


# Weight convergence: training sets of 10/50 CIFAR J8 experiments with different width scalings.
paper_cifar_sizes = [64, 128, 256, 512]
def paper_exp(dataset, js=[], s=1, L=4):
    return base(dataset, grayscale=False) * arch(change_size(paper_cifar_sizes, js, s), L=L, std=True) * classifier_bn("std") * classifier_nobias * no_weight_decay * exp(name="paperexps")
exps |= ors(paper_exp("cifar", js=[1, 2, 3], s=s) for s in [2 ** i for i in range(-3, 6)]) * exp(name="scalewidth") * inits(10)
exps |= iclr("cifar", norm=True) * classifier_bn("std") * classifier_nobias * no_weight_decay * exp(name="paperexps") * inits(50)


# Activation and weight spectra: train a scattering network and ResNet on ImageNet.
exps |= iclr("imagenet", skip=True, complex_p=True)
exps |= resnet(18) * no_biases * batch_norm("post")


# Training dynamics: training one CIFAR J8 experiment with no normalization nor weight decay.
exps |= iclr("cifar", norm=False) * classifier_bn("std") * classifier_nobias * no_weight_decay


exps.run()
