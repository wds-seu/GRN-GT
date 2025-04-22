import pandas as pd
import numpy as np
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=500, help='network scale')
parser.add_argument('--data', type=str, default='hESC', help='data type')
parser.add_argument('--net', type=str, default='STRING', help='network type')
args = parser.parse_args()


def train_val_test_set(label_file, gene_file, tf_file, train_file, val_file, test_file, density):
    label = pd.read_csv(label_file, index_col=0)
    gene_set = pd.read_csv(gene_file, index_col=0)['index'].values
    tf_set = pd.read_csv(tf_file, index_col=0)['index'].values

    tf = label['TF'].values
    tf_list = np.unique(tf)

    # 所有正例
    pos_dict = {}
    for i in tf_list:
        pos_dict[i] = []
    for i, j in label.values:
        pos_dict[i].append(j)

    # 所有负例
    neg_dict = {}
    for i in tf_set:
        neg_dict[i] = []
    for i in tf_set:
        if i in pos_dict.keys():
            pos_item = pos_dict[i]
            pos_item.append(i)
            neg_item = np.setdiff1d(gene_set, pos_item)
            neg_dict[i].extend(neg_item)
            pos_dict[i] = np.setdiff1d(pos_dict[i], i)
        else:
            neg_item = np.setdiff1d(gene_set, i)
            neg_dict[i].extend(neg_item)

    # 将正例划分到训练集、验证集、测试集，比例为3:1:1
    train_pos = {}
    val_pos = {}
    test_pos = {}
    for k in pos_dict.keys():
        train_pos[k] = []
        val_pos[k] = []
        test_pos[k] = []
        for v in pos_dict[k]:
            p = np.random.uniform(0, 1)
            if p < 0.6:
                train_pos[k].append(v)
            elif p < 0.8:
                val_pos[k].append(v)
            else:
                test_pos[k].append(v)

    train_pos_num = sum(len(v) for v in train_pos.values())
    train_neg_num = int(train_pos_num // density - train_pos_num)

    train_neg = {}
    for i in tf_set:
        train_neg[i] = []
    for i in range(train_neg_num):
        t1 = np.random.choice([k for k, v in neg_dict.items() if v])
        t2 = np.random.choice(neg_dict[t1])
        train_neg[t1].append(t2)
        neg_dict[t1].remove(t2)

    print(sum(len(v) for v in train_pos.values()) + sum(len(v) for v in train_neg.values()))
    print(sum(len(v) for v in train_pos.values()))
    print(sum(len(v) for v in train_neg.values()))

    val_pos_num = sum(len(v) for v in val_pos.values())
    val_neg_num = int(val_pos_num // density - val_pos_num)
    val_neg = {}
    for i in tf_set:
        val_neg[i] = []
    for i in range(val_neg_num):
        t1 = np.random.choice([k for k, v in neg_dict.items() if v])
        t2 = np.random.choice(neg_dict[t1])
        val_neg[t1].append(t2)
        neg_dict[t1].remove(t2)

    print(sum(len(v) for v in val_pos.values()))
    print(sum(len(v) for v in val_neg.values()))

    test_pos_num = sum(len(v) for v in test_pos.values())
    test_neg_num = int(test_pos_num // density - test_pos_num)
    test_neg = {}
    for i in tf_set:
        test_neg[i] = []
    for i in range(test_neg_num):
        t1 = np.random.choice([k for k, v in neg_dict.items() if v])
        t2 = np.random.choice(neg_dict[t1])
        test_neg[t1].append(t2)
        neg_dict[t1].remove(t2)

    print(sum(len(v) for v in test_pos.values()))
    print(sum(len(v) for v in test_neg.values()))

    train_pos_set = []
    for k in train_pos.keys():
        for val in train_pos[k]:
            train_pos_set.append([k, val])

    train_neg_set = []
    for k in train_neg.keys():
        for val in train_neg[k]:
            train_neg_set.append([k, val])

    train_set = train_pos_set + train_neg_set
    train_label = [1 for _ in range(len(train_pos_set))] + [0 for _ in range(len(train_neg_set))]
    train_sample = np.array(train_set)
    train = pd.DataFrame()
    train['TF'] = train_sample[:, 0]
    train['Target'] = train_sample[:, 1]
    train['Label'] = train_label
    train.to_csv(train_file)

    val_pos_set = []
    for k in val_pos.keys():
        for val in val_pos[k]:
            val_pos_set.append([k, val])

    val_neg_set = []
    for k in val_neg.keys():
        for val in val_neg[k]:
            val_neg_set.append([k, val])

    val_set = val_pos_set + val_neg_set
    val_label = [1 for _ in range(len(val_pos_set))] + [0 for _ in range(len(val_neg_set))]

    val_sample = np.array(val_set)
    val = pd.DataFrame()
    val['TF'] = val_sample[:, 0]
    val['Target'] = val_sample[:, 1]
    val['Label'] = val_label
    val.to_csv(val_file)

    test_pos_set = []
    for k in test_pos.keys():
        for val in test_pos[k]:
            test_pos_set.append([k, val])

    test_neg_set = []
    for k in test_neg.keys():
        for val in test_neg[k]:
            test_neg_set.append([k, val])

    test_set = test_pos_set + test_neg_set
    test_label = [1 for _ in range(len(test_pos_set))] + [0 for _ in range(len(test_neg_set))]

    test_sample = np.array(test_set)
    test = pd.DataFrame()
    test['TF'] = test_sample[:, 0]
    test['Target'] = test_sample[:, 1]
    test['Label'] = test_label
    test.to_csv(test_file)


def Network_Statistic(data_type, net_scale, net_type):
    if net_type == 'STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165, 'hHEP500': 0.379, 'hHEP1000': 0.377, 'mDC500': 0.085,
               'mDC1000': 0.082, 'mESC500': 0.345, 'mESC1000': 0.347, 'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565, 'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError


if __name__ == '__main__':
    data_type = args.data
    net_type = args.net
    density = Network_Statistic(data_type=data_type, net_scale=args.num, net_type=net_type)
    tf_file = os.getcwd() + '/Dataset/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/TF.csv'
    gene_file = os.getcwd() + '/Dataset/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/Target.csv'
    label_file = os.getcwd() + '/Dataset/' + net_type + ' Dataset/' + data_type + '/TFs+' + str(args.num) + '/Label.csv'
    path = os.getcwd() + '/train_validation_test/' + net_type + '/' + data_type + ' ' + str(args.num)

    if not os.path.exists(path):
        os.makedirs(path)

    train_file = path + '/Train_set.csv'
    test_file = path + '/Test_set.csv'
    val_file = path + '/Validation_set.csv'

    train_val_test_set(label_file, gene_file, tf_file, train_file, val_file, test_file, density)
    print(args.net + " " + args.data + " " + str(args.num) + " finish!")
