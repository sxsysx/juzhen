import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset

# 用于半监督学习的数据包装器，区分有标签和无标签数据
class SemiSupervisedDataset(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset=None):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.has_unlabeled = unlabeled_dataset is not None
        
    def __len__(self):
        labeled_len = len(self.labeled_dataset)
        if self.has_unlabeled:
            # 有标签数据和无标签数据的总长度
            # 注意：实际训练时我们会分别迭代有标签和无标签数据
            return labeled_len + len(self.unlabeled_dataset)
        return labeled_len
    
    def __getitem__(self, index):
        # 主要用于支持常规的数据加载器功能，实际半监督训练会有专门的处理
        if index < len(self.labeled_dataset):
            # 有标签数据，返回图像和标签
            img, target = self.labeled_dataset[index]
            return img, target, True  # 第三个参数表示是否有标签
        else:
            # 无标签数据，返回图像和-1作为伪标签
            img, _ = self.unlabeled_dataset[index - len(self.labeled_dataset)]
            return img, -1, False

# 获取有标签和无标签数据加载器
def get_semisupervised_data_loaders(args):
    # 获取基本的数据转换器
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.ngpu else {}
    
    # 创建训练数据转换器
    if args.raw_data:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        if not args.noaug:
            # 带数据增强的转换器
            transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # 无数据增强的转换器
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    
    # 一致性正则化需要的弱增强和强增强
    transform_weak = transform_train  # 弱增强使用常规训练的增强
    
    # 强增强（用于一致性正则化）
    transform_strong = transforms.Compose([
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])
    
    # 加载有标签数据
    labeled_dataset = torchvision.datasets.STL10(root=args.data_dir, split='train', download=True,
                                               transform=transform_weak)
    testset = torchvision.datasets.STL10(root=args.data_dir, split='test', download=True,
                                      transform=transform_test)
    # 对于STL10，我们可以使用split='unlabeled'加载无标签数据
    if args.use_semisupervised:
        unlabeled_dataset = torchvision.datasets.STL10(root=args.data_dir, split='unlabeled', download=True,
                                                      transform=transform_weak)
    else:
        unlabeled_dataset = None
    
    # 创建测试数据加载器
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)
    
    if args.use_semisupervised:
        # 创建有标签和无标签数据的加载器
        labeled_loader = torch.utils.data.DataLoader(
            labeled_dataset,
            batch_size=args.labeled_batch_size if hasattr(args, 'labeled_batch_size') else args.batch_size,
            shuffle=True,
            **kwargs
        )
        
        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_dataset,
            batch_size=args.unlabeled_batch_size if hasattr(args, 'unlabeled_batch_size') else args.batch_size,
            shuffle=True,
            **kwargs
        )
        
        # 也创建一个半监督数据集用于某些需要的情况
        semi_dataset = SemiSupervisedDataset(labeled_dataset, unlabeled_dataset)
        
        return labeled_loader, unlabeled_loader, testloader, transform_weak, transform_strong
    else:
        # 常规的监督学习数据加载器
        trainloader = torch.utils.data.DataLoader(labeled_dataset, batch_size=args.batch_size,
                                                  shuffle=True, **kwargs)
        return trainloader, None, testloader, transform_weak, transform_strong

def get_data_loaders(args):
    """原始的数据加载器函数，保持向后兼容"""
    # 如果使用半监督学习，调用专门的函数
    if hasattr(args, 'use_semisupervised') and args.use_semisupervised:
        labeled_loader, unlabeled_loader, testloader, _, _ = get_semisupervised_data_loaders(args)
        # 为了向后兼容，返回有标签数据加载器作为trainloader
        return labeled_loader, testloader
    
    # 原始的实现
    if args.trainloader and args.testloader:
        assert os.path.exists(args.trainloader), 'trainloader does not exist'
        assert os.path.exists(args.testloader), 'testloader does not exist'
        trainloader = torch.load(args.trainloader)
        testloader = torch.load(args.testloader)
        return trainloader, testloader
    
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.ngpu else {}

    #print('STL10 is on the test')
    if args.raw_data:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        if not args.noaug:
            # with data augmentation
            transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # no data agumentation
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])


    trainset = torchvision.datasets.STL10(root=args.data_dir, split='train', download=True,
                                        transform=transform_train)
    testset = torchvision.datasets.STL10(root=args.data_dir, split='test', download=True,
                                       transform=transform_test)    

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)



    return trainloader, testloader
