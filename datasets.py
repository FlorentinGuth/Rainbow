import os

import torch.utils.data.dataset
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *


def get_datasets(dataset_name: str, grayscale: bool, imagenet_path: str = "",
                 image_size=None, randomize_label_seed=None):
    """ Returns train and validation datasets. Can optionally resize images or randomize labels with a given seed. """
    # Train-only transforms (data augmentation) and test-only transforms (center crop for ImageNet).
    train_transforms = []
    test_transforms = []
    if dataset_name == "ImageNet":
        if image_size is None:
            image_size = 224
        train_transforms = [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
        test_transforms = [transforms.Resize((256 * image_size) // 224), transforms.CenterCrop(image_size)]
    elif dataset_name.startswith("CIFAR") or dataset_name == "ImageNet32":
        if image_size is None:
            image_size = 32
        train_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(image_size, padding=4)]

    # Common transforms: normalization and grayscale.
    common_transforms = []
    if dataset_name == "MNIST":
        mean = [0.1307]
        std = [0.3081]
    else:
        if grayscale:
            mean = [0.481]
            std = [0.239]
            common_transforms.append(transforms.Grayscale())
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
    common_transforms.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # Target transforms (to the labels).
    target_transforms = []
    if dataset_name == "ImageNet32":
        target_transforms = [transforms.Lambda(lambda y: y - 1)]  # Remove 1 to labels to put them in the range 0..999.

    dataset_class = dict(
        CIFAR10=datasets.CIFAR10, CIFAR100=datasets.CIFAR100, MNIST=datasets.MNIST,
        ImageNet=datasets.ImageFolder, ImageNet32=ImageNet32,
    )[dataset_name]

    def get_dataset(train: bool):  # Returns the train or validation dataset.
        if dataset_name == "ImageNet":
            root = os.path.join(imagenet_path, "train" if train else "val")
        else:
            root = "./data"

        kwargs = dict(
            root=root,
            transform=transforms.Compose((train_transforms if train else test_transforms) + common_transforms),
            target_transform=transforms.Compose(target_transforms),
        )
        if dataset_name != "ImageNet":
            kwargs.update(train=train, download=True)

        dataset = dataset_class(**kwargs)
        if randomize_label_seed is not None:
            num_classes = 1000 if "ImageNet" in dataset_name else 100 if dataset_name == "CIFAR100" else 10
            dataset = RandomLabelsDataset(dataset, num_classes=num_classes, seed=randomize_label_seed)
        return dataset

    train_dataset = get_dataset(train=True)
    val_dataset = get_dataset(train=False)
    return train_dataset, val_dataset


def take_subset(train_dataset, val_dataset,
                classes_subset: Optional[Iterable[int]] = None, data_subset: Optional[Iterable[int]] = None):
    # Can use a subset of all classes for ImageNet (specified in a file or randomly chosen).
    if classes_subset is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, [i for i, y in enumerate(train_dataset.targets) if y in classes_subset])
        val_dataset = torch.utils.data.Subset(
            val_dataset, [i for i, y in enumerate(val_dataset.targets) if y in classes_subset])

    if data_subset is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, data_subset)

    return train_dataset, val_dataset


def get_dataloaders(args, logfile, summaryfile):
    """ Returns train and validation data loaders. Takes care of taking a subset of the data if required. """
    dataset_name = args.dataset
    print_and_write(f"Working on {dataset_name}", logfile, summaryfile)

    train_dataset, val_dataset = get_datasets(
        dataset_name=dataset_name, grayscale=args.grayscale, imagenet_path=args.data,
        image_size=args.resize_images, randomize_label_seed=args.randomize_labels,
    )

    train_dataset, val_dataset = take_subset(train_dataset=train_dataset, val_dataset=val_dataset,
                                             classes_subset=args.classes_subset, data_subset=args.data_subset)

    def get_dataloader(dataset, shuffle):
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                           num_workers=args.workers, pin_memory=True)

    train_loader = get_dataloader(train_dataset, shuffle=True)
    val_loader = get_dataloader(val_dataset, shuffle=False)
    return train_loader, val_loader


class ImageNet32(torchvision.datasets.CIFAR10):
    """ ImageNet32 dataset. """

    base_folder = "imagenet32"
    url = None
    filename = None
    tgz_md5 = None
    train_list = [
        ["train_data_batch_1",  "6c4495e65cd24a8c3019857ef9b758ee"],
        ["train_data_batch_2",  "3dd727de4155836807dfc19f98c9fe70"],
        ["train_data_batch_3",  "03d3dc4e850e23e1d526f268a0196296"],
        ["train_data_batch_4",  "876f7e6d6ddb1f52ecb654ffdc8293f6"],
        ["train_data_batch_5",  "c789bcdd1feed34a9bc58598a1a1bf7d"],
        ["train_data_batch_6",  "8ce5344cb1e11f31bc507cae4c856083"],
        ["train_data_batch_7",  "32ecc8ad6c55b1c9cb6cf79a2cc46094"],
        ["train_data_batch_8",  "bdeb6da3d05771121992b48c59c69439"],
        ["train_data_batch_9",  "58417149b5ce31688f805341e7f57b4f"],
        ["train_data_batch_10", "46ad60a1144aaf97a143914453b0dabb"],
    ]

    test_list = [
        ["val_data", "06c02b8b4c8de8b3a36b07859a49de6f"],
    ]

    meta = {}

    def _load_meta(self):
        pass


class RandomLabelsDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, original_dataset, num_classes, seed=None):
        self.original_dataset = original_dataset
        self.size = len(self.original_dataset)

        # Generate random labels.
        random_state = np.random.RandomState(seed)
        self.labels = random_state.randint(low=0, high=num_classes, size=self.size)

    def __getitem__(self, item):
        # Throw away true label and return a fake one.
        x, _ = self.original_dataset[item]
        y = self.labels[item]
        return x, y

    def __len__(self):
        return self.size
