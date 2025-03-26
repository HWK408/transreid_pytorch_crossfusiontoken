import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, RandomIdentitySampler_IdUniform
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .mm import MM
from .cross_modality_datasets import SYSUDataset, RegDBDataset
from .sampler2 import RandomIdentitySampler, CrossModalityIdentitySampler, NormTripletSampler
from prettytable import PrettyTable
import logging

__factory = {
    'market1501': Market1501,
    'msmt17': MSMT17,
    'mm': MM,
    'sysu': SYSUDataset,
    'regdb': RegDBDataset,
}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

# def collate_fn(batch):  # img, label, cam_id, img_path, img_id
#     samples = list(zip(*batch))
#     data = [torch.stack(x, 0) for i, x in enumerate(samples)]
#     return data

def collate_fn(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))

    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i != 3]
    data.insert(3, samples[3])
    return data

def collate_fn_test(batch):  # img, label, cam_id, img_path, img_id
    samples = list(zip(*batch))
    data = [torch.stack(x, 0) for i, x in enumerate(samples) if i !=1 and i!=3]
    data.insert(1, samples[1])
    data.insert(3, samples[3])
    return data

def show_dataset_info(dataset, query, gallery):
    logger = logging.getLogger("transreid.dataset")
    num_train_pids, num_train_imgs, num_train_captions = dataset.num_ids, len(dataset.img_paths)-dataset.cam_ids.count(3)-dataset.cam_ids.count(6), dataset.cam_ids.count(3)+dataset.cam_ids.count(6)
    num_test_pids, num_test_imgs, num_test_captions = query.num_ids, len(query.img_paths), len(
            gallery.img_paths)

    # TODO use prettytable print comand line table

    logger.info(f"{dataset.__class__.__name__} Dataset statistics:")
    table = PrettyTable(['subset', 'ids', 'visible', 'infrared'])
    table.add_row(
        ['train', num_train_pids, num_train_imgs, num_train_captions])
    table.add_row(
        ['query', num_test_pids, num_test_imgs, num_test_captions])
    logger.info('\n' + str(table))

def make_dataloader(cfg):
    global train_dataset, query_dataset, gallery_dataset
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    batch_size = cfg.SOLVER.IMS_PER_BATCH
    p_size = 16
    k_size = 8
    sample_method = 'identity_random'
    if cfg.DATASETS.NAMES == 'sysu':
        train_dataset = SYSUDataset(root=cfg.DATASETS.ROOT_DIR, mode='train', transform=train_transforms)
        gallery_dataset = SYSUDataset(root=cfg.DATASETS.ROOT_DIR, mode='gallery', transform=val_transforms)
        query_dataset = SYSUDataset(root=cfg.DATASETS.ROOT_DIR, mode='query', transform=val_transforms)
    elif cfg.DATASETS.NAMES == 'regdb':
        train_dataset = RegDBDataset(root=cfg.DATASETS.ROOT_DIR, mode='train', transform=train_transforms)
        gallery_dataset = RegDBDataset(root=cfg.DATASETS.ROOT_DIR, mode='gallery', transform=val_transforms)
        query_dataset = RegDBDataset(root=cfg.DATASETS.ROOT_DIR, mode='query', transform=val_transforms)
    
    show_dataset_info(train_dataset, query_dataset, gallery_dataset)

    # sampler
    assert sample_method in ['random', 'identity_uniform', 'identity_random', 'norm_triplet']
    if sample_method == 'identity_uniform':
        batch_size = p_size * k_size
        sampler = CrossModalityIdentitySampler(train_dataset, p_size, k_size)
    elif sample_method == 'identity_random':
        batch_size = p_size * k_size
        sampler = RandomIdentitySampler(train_dataset, p_size * k_size, k_size)
    elif sample_method == 'norm_triplet':
        batch_size = p_size * k_size
        sampler = NormTripletSampler(train_dataset, p_size * k_size, k_size)
    else:
        sampler = CrossModalityRandomSampler(train_dataset, batch_size)
    # loader
    train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, drop_last=True, pin_memory=True,
                              collate_fn=collate_fn, num_workers=num_workers)

    # dataloader
    query_loader = DataLoader(dataset=query_dataset,
                              batch_size=cfg.TEST.IMS_PER_BATCH,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=collate_fn,
                              num_workers=num_workers)

    gallery_loader = DataLoader(dataset=gallery_dataset,
                                batch_size=cfg.TEST.IMS_PER_BATCH,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                collate_fn=collate_fn,
                                num_workers=num_workers)

    num_classes = train_dataset.num_ids
    cam_num = len(set(train_dataset.cam_ids))
    view_num = 2

    return train_loader, query_loader, gallery_loader, len(query_dataset.img_paths), num_classes, cam_num, view_num
