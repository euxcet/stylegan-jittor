import math
import random
import argparse
import numpy as np
from PIL import Image

import jittor as jt
from model import StyledGenerator, Discriminator


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument( '--phase', type=int, default=600_000, help='number of samples used for each training phases',)
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument( '--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument( '--no_from_rgb_activate', action='store_true', help='use activate in from_rgb (original implementation)',)
    parser.add_argument( '--mixing', action='store_true', help='use mixing regularization')
    parser.add_argument( '--loss', type=str, default='wgan-gp', choices=['wgan-gp', 'r1'], help='class of gan loss',)
    args = parser.parse_args()

    generator = StyledGenerator(code_size)
    discriminator = Discriminator(from_rgb_activate=not args.no_from_rgb_activate)
