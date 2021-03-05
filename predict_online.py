# coding: UTF-8
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
import pandas as pd
from sklearn import metrics
from utils import build_dataset, build_iterator, build_query, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
args = parser.parse_args()

def predict(config, model2, model135, query):
    # out model predict result for analysis
    start_time = time.time()

    input = build_query(config, query)
    with torch.no_grad():
        outputs2 = model2(input)
        outputs135 = model135(input)
        # scl_loss = scl(outputs, labels)
        # loss = torch.tensor([cross_loss, scl_loss]) * model.loss_weight
        result2 = torch.max(outputs2.data, 1)[1].cpu().numpy()
        result135 = torch.max(outputs135.data, 1)[1].cpu().numpy()
    print('bi class:{}'.format(result2))
    print('bi class:{}'.format(outputs2.data))
    print('multi class {}'.format(result135))
    print('multi class {}'.format(torch.max(outputs135.data, 1)[0]))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def loadModel(dataset):
    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path))
    return model, config 



if __name__ == '__main__':
    dataset2 = 'data/Intention2'  # 数据集
    dataset135 = 'data/Intention135'
    model2, config2 = loadModel(dataset2)
    model135, config135 = loadModel(dataset135)

    prompt = "\nPlease enter the query(Enter 'quit' when you are finished.) :\n"
    while True:
        query = input(prompt)
        if query == 'quit':
            break
        else:
            print("input query:" + query + "\n")
            predict(config2, model2, model135, query)
    
