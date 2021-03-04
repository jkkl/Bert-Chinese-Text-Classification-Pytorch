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

def predict(config, model, test_iter, out_file):
    # out model predict result for analysis
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, out_file, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, out_file, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # scl = SCLLoss(2).to(config.device)
    # out_wr = open(out_file, 'w')
    result_pd = pd.DataFrame()
    with torch.no_grad():
        for texts, labels, queries in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            # scl_loss = scl(outputs, labels)
            # loss = torch.tensor([cross_loss, scl_loss]) * model.loss_weight
            loss_total += torch.sum(loss)
            labels = labels.data.cpu().numpy()
            predicts = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predicts)
            # save result
            is_diff = [label == pred for label, pred in zip(labels, predicts)]
            batch_out_data = {'query': queries, 'label': labels, 'predict': predicts, 'is_diff': is_diff}
            batch_result = pd.DataFrame(batch_out_data)
            result_pd = result_pd.append(batch_result)

    result_pd.to_csv(out_file, sep='\t')
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

if __name__ == '__main__':
    # dataset = 'data/Intention2_V2'  # 数据集
    dataset = 'data/Intention135'  # 数据集

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

    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    # train(config, model, train_iter, dev_iter, test_iter)
    
    out_result_file = dataset + '/result/model_name.result.txt'
    predict(config, model, test_iter, out_file=out_result_file)
