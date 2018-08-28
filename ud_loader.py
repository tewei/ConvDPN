# -*- coding: utf-8 -*-
__author__ = "Alexander Shieh (b05401009@ntu.edu.tw)"
import re
import sys
import csv
import math
import json
import sys
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from scipy import spatial
from operator import itemgetter
csv.field_size_limit(sys.maxsize)

def sigmoid(x, alpha=100.0):
    return 1.0/(1.0+np.exp(-1.0*alpha*x))

def read_ud_file(filename='UD_data/zh_gsd-ud-train.conllu'):
    sent_set = []
    sent_idx = -1
    #tmp_cnt = 0
    with open(filename, 'r') as fp:
        for row in fp:
            row = re.split('\t|\n', row)
            if len(row) <= 2:
                continue
            elif row[0] == '1':
                sent_idx += 1
                sent_set.append([])
                sent_set[sent_idx].append(['#', '#', 0, 'NULL'])
                try:
                    sent_set[sent_idx].append([row[1], row[3], int(row[6]), row[7]])
                    #if(row[7] == 'obj'):
                    #    tmp_cnt += 1
                except:
                    print(row)
                    continue
            else:
                try:
                    sent_set[sent_idx].append([row[1], row[3], int(row[6]), row[7]])
                    #if(row[7] == 'obj'):
                    #    tmp_cnt += 1
                except:
                    print(row)
                    continue
    #print(tmp_cnt)
    print(sent_set[-1])
    return sent_set

def main():
    sent_set = read_ud_file()
    dpn_dict = {}
    for sent in sent_set:
        for p in sent:
            if p[3] not in dpn_dict:
                dpn_dict[p[3]] = []
            else:
                try:
                    dpn_dict[p[3]].append((sent[p[2]][0], p[0]))
                except:
                    continue

    for k in dpn_dict:
        print(k,len(dpn_dict[k]))
    
if __name__ == "__main__":
    main()
