# coding: UTF-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import torch
import numpy as np
from train_eval import *
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--is_warmup', type=str, required=True, help='is warm up : True or False')
parser.add_argument('--data_set', type=str, required=True, help='Data set directory')
parser.add_argument('--loss_func', type=str, required=True, help='loss func: ce or ce + scl')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.data_set  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    if args.is_warmup:
        train_data, dev_data, test_data = train_data[:200], dev_data[:100], test_data[:100]
        config.batch_size = 16
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    # test(config, model, test_iter)
    train(config, model, train_iter, dev_iter, test_iter)
    predict(config, model, test_iter)
