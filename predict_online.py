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

def predict(config, model, query):
    # out model predict result for analysis
    model.load_state_dict(torch.load(config.save_path))
    start_time = time.time()

    input = build_query(config, query)
    with torch.no_grad():
        outputs = model(input)
        # scl_loss = scl(outputs, labels)
        # loss = torch.tensor([cross_loss, scl_loss]) * model.loss_weight
        result = torch.max(outputs.data, 1)[1].cpu().numpy()
    print(result)
    print(outputs.data)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    dataset = 'data/Intention2'  # 数据集

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    prompt = "\nPlease enter the query(Enter 'quit' when you are finished.) :\n"
    while True:
        query = input(prompt)
        if query == 'quit':
            break
        else:
            print("input query:" + query + "\n")
            predict(config, model, query)
    
