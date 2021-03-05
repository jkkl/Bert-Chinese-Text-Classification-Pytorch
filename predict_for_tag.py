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
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='bert', help='choose a model: Bert, ERNIE')
args = parser.parse_args()


def predict_tag(config_list, model_list, data_iter, out_file):
    result_pd = pd.DataFrame()
    with torch.no_grad():
        for texts, labels, queries in data_iter:
            outputs = []
            for config, model in zip(config_list, model_list):
                batch_output = model(texts)
                batch_labels = torch.max(batch_output.data, 1)[1].cpu().numpy()
                batch_labels_name = [config.class_list[i] for i in batch_labels]
                batch_scores = torch.max(torch.softmax(batch_output.data, 1),1)[0].cpu().numpy()
                outputs.append((batch_labels, batch_scores, batch_labels_name))
            
            is_diff = [(label == pred).cpu().numpy() for label, pred in zip(labels, outputs[0][0])]
            batch_out_data = {'query': queries, 'origin_bi_label': labels.cpu().numpy(), 'predict_bi_label': outputs[0][0],'predict_bi_score': outputs[0][1],
                              'is_diff': is_diff, "predict_mu_label": outputs[1][2], "predict_mu_score": outputs[1][1]}
            batch_result = pd.DataFrame(batch_out_data)
            result_pd = result_pd.append(batch_result)
    result_pd.to_csv(out_file, sep='\t')

def load_model_dict(dataset_list, model_type):
    model_list = []
    config_list = []

    for dataset in dataset_list:
        x = import_module('models.' + model_type)
        config = x.Config(dataset)
        model = x.Model(config).to(config.device)
        model.load_state_dict(torch.load(config.save_path))
        model.eval()
        model_list.append(model)
        config_list.append(config)
    
    return model_list, config_list

def load_tag_data(config, data_type):

    train_data, dev_data, test_data = build_dataset(config)
    if data_type == 'train':
        tag_data = build_iterator(train_data, config)
    elif data_type == 'dev':
        tag_data = build_iterator(dev_data, config)
    elif data_type == 'test':
        tag_data = build_iterator(test_data, config)
    
    return tag_data


if __name__ == '__main__':
    dataset2 = 'data/Intention2_V2'  # 数据集
    dataset135 = 'data/Intention135'  # 数据集
    dataset_list = [dataset2, dataset135]
    model_type = args.model  # bert
    model_list, config_list = load_model_dict(dataset_list, model_type)
    x = import_module('models.' + model_type)
    tag_data_config = x.Config(dataset2)
    data_type = 'dev'
    tag_data = load_tag_data(tag_data_config, data_type)

    out_file = dataset2 + f'/iter/{data_type}_iter.csv'
    predict_tag(config_list, model_list, tag_data, out_file)
