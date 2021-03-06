# coding: UTF-8
import os
import time
import torch
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

from tools.logger import logger, setting_logging


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
parser.add_argument('--data_dir', type=str, help='data_dir, eg: data/Intention2')
parser.add_argument('--task_name', type=str, help='task_name used in log name and sample file prefix')
parser.add_argument('--task_desc', type=str, help='task_desc used in log name ')
args = parser.parse_args()


if __name__ == '__main__':
    task_name = args.task_name
    task_desc = args.task_desc
    setting_logging("{}_{}".format(task_name, task_desc))
    # logger.info()
    
    dataset = args.data_dir  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset, task_name)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    logger.info("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_data, dev_data, test_data = train_data, dev_data, test_data
    config.batch_size = 64
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    logger.info("Time usage:".format(time_dif))

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
    test(config, model, test_iter)
    logger.info("Game Over !!!")
