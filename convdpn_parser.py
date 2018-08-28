# -*- coding: utf-8 -*-
__author__ = "Alexander Shieh (b05401009@ntu.edu.tw)"

import numpy as np
from scipy import spatial
from ud_loader import read_ud_file
import csv
from convdpn import *
from edmonds import *
from collections import defaultdict
from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = ArgumentParser()
parser.add_argument("-i", "--input-file", help="optional argument", dest="input", default="default")
parser.add_argument("-o", "--output-file", help="optional argument", dest="output", default="default")
args = parser.parse_args()
INPUT_FILE = args.input
OUTPUT_FILE = args.output

class ConvDPNP(object):
    def __init__(self, label_model_filename='convdpn_label_zh_model.h5', edge_model_filename='convdpn_edge_zh_model.h5', word2vec_filename='skip-gram/word2vec.model', max_len=111):
        self.label_model = ConvDPN(word2vec_filename, mode='label')
        self.label_model.max_len = max_len
        self.label_model.create_model()
        self.edge_model = ConvDPN(word2vec_filename, mode='edge')
        self.edge_model.max_len = max_len
        self.edge_model.create_model()
        self.label_model.load_model(label_model_filename)
        self.edge_model.load_model(edge_model_filename)
        self.w2v = Word2Vec.load(word2vec_filename)
        

    def raw_score(self, w1, w2):
        try:
            t1 = self.w2v.wv[w1]
            c2 = self.w2v.syn1neg[self.w2v.wv.vocab[w2].index]
            t2 = self.w2v.wv[w2]
            c1 = self.w2v.syn1neg[self.w2v.wv.vocab[w1].index]
            score = 1-spatial.distance.cosine(t2, c1)
            return score
        except:
            return 0.0

    def gen_tree(self, edge_graph, label_graph, label_graph_p, sent_tok, sent_pos):
        G1 = defaultdict(dict)
        for e in label_graph:
            G1[e[0]][e[1]] = e[2]
        G2 = defaultdict(dict)
        for e in edge_graph:
            G2[e[0]][e[1]] = -e[2]

        return G1, G2

    def eval_tree(self, root, G):
        p = 0
        G3 = mst(root, G)
        for j in G3:
            for k in G3[j]:
                p += G[j][k]

        return p, G3        

    def parse(self, sent_tok, sent_pos):
        label_graph, mat, label_graph_p = self.label_model.test_model(sent_tok, sent_pos) 
        edge_graph, mat = self.edge_model.test_model(sent_tok, sent_pos)
        G1, G2 = self.gen_tree(edge_graph, label_graph, label_graph_p, sent_tok, sent_pos)
        root = -1
        G3 = defaultdict(dict)
        G4 = defaultdict(dict)
        max_prob = 1e9
        
        root_list = range(len(sent_tok))
        root_list = sorted(root_list, key= lambda x: G2[x][x])
        root_p = [G2[x][x] for x in range(len(sent_tok))]
        
        for i in root_list[:1]:
            p, G3_tmp = self.eval_tree(i, G2)
            if p < max_prob:
                root = i
                max_prob = p
                G3 = G3_tmp
        
        for j in G3:
            for k in G3[j]:
                G4[j][k] = G1[j][k]

        return root, G4

    def tree_to_labels(self, sent_tok, sent_pos, root, G):
        labels = [['ROOT', '#', 0, 'NULL'] for i in range(len(sent_tok)+1)]
        for j in G:
            for k in G[j]:
                labels[k+1] = [sent_tok[k], sent_pos[k], j+1, G[j][k]]
        
        labels[root+1] = [sent_tok[root], sent_pos[root], 0, 'root']
        
        for i in range(len(sent_pos)):
            if(sent_pos[i] == 'PUNCT'):
                labels[i+1] = [sent_tok[i], sent_pos[i], root+1, 'punct']

        return labels

def eval():
    Parser = ConvDPNP()
    ud_dev_set = read_ud_file('UD_data/zh_gsd-ud-dev.conllu')
    print(ud_dev_set[0])
    UAS = 0
    LAS = 0
    RAS = 0
    num_edges = 0
    for s in ud_dev_set:
        sent_tok = [p[0] for p in s[1:]]
        sent_pos = [p[1] for p in s[1:]]
        
        root_real = -1
        for i in range(len(s)):
            if s[i][3] == 'root':
                root_real = i-1
        
        root, G = Parser.parse(sent_tok, sent_pos)
        
        if(root == root_real):
            RAS +=1

        res = Parser.tree_to_labels(sent_tok, sent_pos, root, G)

        if(len(res) != len(s)):
            print('ERROR!')
        for i in range(1, len(s)):
            num_edges += 1

            print(res[i], '\t', s[i])
            if(res[i][2] == int(s[i][2])):
                UAS += 1
                if(res[i][3] == s[i][3]):
                    LAS += 1

    print('UAS = '+str(UAS/num_edges))
    print('LAS = '+str(LAS/num_edges))
    print('RAS = '+str(RAS/len(ud_dev_set)))

def test():
    Parser = ConvDPNP()
    test_set = []
    with open(INPUT_FILE, 'r') as fx:
        reader = csv.reader(fx)
        test_set = list(reader)

    with open(OUTPUT_FILE, 'w') as fy:
        wr = csv.writer(fy)

        for s in test_set:
            sent_tok = s[0].split(' ')
            sent_pos = s[1].split(' ')
            root, G = Parser.parse(sent_tok, sent_pos)
            res = Parser.tree_to_labels(sent_tok, sent_pos, root, G)
            wr.writerows(res)
    
def main():
    #eval()
    test()
            
    
if __name__ == "__main__":
    main()


