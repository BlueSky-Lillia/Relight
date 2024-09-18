from share import *
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
# from tutorial_dataset import MyDataset
from dataset import RelightDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os


def args_parser():
    # Configs
    parser = argparse.ArgumentParser('Training relighting diffusion', add_help=False)
    parser.add_argument('--learning_rate', default=5*1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--logger_freq', default=300, type=int)
    parser.add_argument('--sd_locked', action='store_true', default=True)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--resume_path', default='./models/control_sd15.ckpt', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    return parser

import torch
torch.cuda.empty_cache()

if __name__ =='__main__': # which is important to multiprocess

    parser = args_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    batch_size = args.batch_size
    logger_freq = args.logger_freq

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    if os.path.exists(args.resume_path):
        model_weight = load_state_dict(args.resume_path, location='cpu')
        model.load_state_dict(model_weight, strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control

    # Misc
    dataset = RelightDataset(root = '/home/wangzhen/Data/Raw/ppr10k')
    dataset.display2(random_index=0)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)
