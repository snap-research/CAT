import torch

from metric.cityscapes_mIoU import DRNSeg
from metric.deeplabv2 import DeepLabV2, MSC
from metric.fid_score import _compute_statistics_of_ims, calculate_frechet_distance
from metric.inception import InceptionV3
from utils import util
from .mIoU_score import test


def get_fid(fakes, model, npz, device, batch_size=1, use_tqdm=True):
    m1, s1 = npz['mu'], npz['sigma']
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes).astype(float)
    m2, s2 = _compute_statistics_of_ims(fakes,
                                        model,
                                        batch_size,
                                        2048,
                                        device,
                                        use_tqdm=use_tqdm)
    return float(calculate_frechet_distance(m1, s1, m2, s2))


def get_mIoU(fakes,
             names,
             model,
             device,
             table_path='datasets/table.txt',
             data_dir='database/cityscapes',
             batch_size=1,
             num_workers=8,
             num_classes=19,
             use_tqdm=True):
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes)
    mAP = test(fakes,
               names,
               model,
               device,
               table_path=table_path,
               data_dir=data_dir,
               batch_size=batch_size,
               num_workers=num_workers,
               num_classes=num_classes,
               use_tqdm=use_tqdm)
    return float(mAP)


def get_cityscapes_mIoU(fakes,
                        names,
                        model,
                        device,
                        table_path='datasets/table.txt',
                        data_dir='database/cityscapes',
                        batch_size=1,
                        num_workers=8,
                        num_classes=19,
                        tqdm_position=None):
    from .cityscapes_mIoU import test
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes)
    mIoU = test(fakes,
                names,
                model,
                device,
                table_path=table_path,
                data_dir=data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                num_classes=num_classes,
                tqdm_position=tqdm_position)
    return float(mIoU)


def get_fid_new(reals, fakes, model, device, batch_size=1, use_tqdm=True):
    reals = torch.cat(reals, dim=0)
    reals = util.tensor2im(reals).astype(float)
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes).astype(float)

    m1, s1 = _compute_statistics_of_ims(reals,
                                        model,
                                        batch_size,
                                        2048,
                                        device,
                                        use_tqdm=use_tqdm,
                                        median=False)
    m2, s2 = _compute_statistics_of_ims(fakes,
                                        model,
                                        batch_size,
                                        2048,
                                        device,
                                        use_tqdm=use_tqdm,
                                        median=False)
    fid_mean = float(calculate_frechet_distance(m1, s1, m2, s2, median=False))

    m1, s1 = _compute_statistics_of_ims(reals,
                                        model,
                                        batch_size,
                                        2048,
                                        device,
                                        use_tqdm=use_tqdm,
                                        median=True)
    m2, s2 = _compute_statistics_of_ims(fakes,
                                        model,
                                        batch_size,
                                        2048,
                                        device,
                                        use_tqdm=use_tqdm,
                                        median=True)
    fid_median = float(calculate_frechet_distance(m1, s1, m2, s2, median=True))

    return fid_mean, fid_median


def get_mIoU_new(fakes,
                 names,
                 model,
                 device,
                 table_path='datasets/table.txt',
                 data_dir='database/cityscapes',
                 batch_size=1,
                 num_workers=8,
                 num_classes=19,
                 use_tqdm=True):
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes)
    mAP, medianAP = test(fakes,
                         names,
                         model,
                         device,
                         table_path=table_path,
                         data_dir=data_dir,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         num_classes=num_classes,
                         use_tqdm=use_tqdm,
                         median=True)
    return float(mAP), float(medianAP)
