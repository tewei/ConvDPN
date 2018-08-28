# -*- coding: utf-8 -*-
__author__ = "Alexander Shieh (b05401009@ntu.edu.tw)"
import math
import random
import json
import re
import random
import itertools
import numpy as np
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Dot, Multiply, Average, Concatenate, Flatten
from keras.layers import Lambda, Reshape, MaxPool2D, Conv2D, Dropout, Conv1D, MaxPool1D
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from ud_loader import read_ud_file

class ConvDPN(object):
    def __init__(self, word2vec_filename, mode='label'):
        self.w2v = Word2Vec.load(word2vec_filename)
        self.vocab_size = len(self.w2v.wv.vocab)
        #remove punct, root
        # self.label_list = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'case:loc', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'discourse:sp', 'dislocated', 'fixed', 'flat', 'flat:name', 'iobj', 'mark', 'mark:adv', 'mark:prt', 'mark:relcl', 'nmod', 'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:agent', 'obl:patient', 'obl:tmod', 'parataxis', 'vocative', 'xcomp']
        self.label_list = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:caus', 'aux:pass', 'case', 'case:aspect', 'case:dec', 'case:pref', 'case:suff', 'cc', 'ccomp', 'clf', 'conj', 'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'discourse', 'dislocated', 'flat:foreign', 'iobj', 'mark', 'mark:advb', 'mark:comp', 'mark:relcl', 'nmod', 'nmod:tmod', 'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'orphan', 'vocative', 'xcomp']
        self.label_dict = {}
        for i in range(len(self.label_list)):
            self.label_dict[self.label_list[i]] = i
        self.num_labels = len(self.label_list)
        self.pos_list = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SYM', 'SCONJ', 'VERB', 'X']
        self.pos_dict = {}
        for i in range(len(self.pos_list)):
            self.pos_dict[self.pos_list[i]] = i

        self.unk_idx = self.w2v.wv.syn0.shape[1]
        self.mode = mode
        self.seq_corpus = []
        self.seq_len = []
        self.seq_pos = []
        self.seq_edges = []
        self.max_len = -1

    def load_train_data(self, corpus, train=True):
        num_s = -1
        self.max_len = max( (len(s)-1) for s in corpus)

        for s in corpus:
            num_s += 1
            if train == True:
                self.seq_corpus.append([])
                self.seq_pos.append([])
                self.seq_edges.append([])
                self.seq_len.append(len(s))
                for i in range(1, len(s)):
                    if(s[i][0] in self.w2v.wv):
                        self.seq_corpus[num_s].append(self.w2v.wv.vocab[s[i][0]].index)
                    else:
                        self.seq_corpus[num_s].append(self.unk_idx)
                    if(s[i][2] == 0):
                        self.seq_edges[num_s].append((i-1, i-1, -1))
                    elif(s[i][3] in self.label_dict):
                        self.seq_edges[num_s].append((s[i][2]-1, i-1, self.label_dict[s[i][3]]))
                    self.seq_pos[num_s].append(self.pos_dict[s[i][1]])

                self.seq_corpus[num_s] += [0]*(self.max_len-len(s)+1)
                self.seq_pos[num_s] += [0]*(self.max_len-len(s)+1)
                    
        #print(self.seq_corpus[0])
        #print(self.seq_edges[0])
        #print(self.seq_pos[0])
        #print(len(self.seq_corpus))
        #print(len(self.seq_edges))
    
    def gen_train_data(self):
        X_train_words = []
        X_train_out = []
        X_train_pos = []
        X_train_in = []
        Y_train = []
        #W_train = []
        #i_list = random.sample(range(len(self.seq_corpus)), sample_size)

        if (self.mode == 'edge'):
            for i in range(len(self.seq_corpus)):

                real_tup = []
                for e in self.seq_edges[i]:
                    #Real Samples
                    X_train_words.append(self.seq_corpus[i])
                    X_train_out.append(e[0])
                    X_train_in.append(e[1])
                    X_train_pos.append(self.seq_pos[i])
                    Y_train.append(1)
                    real_tup.append((e[0], e[1]))

                    # #Negative Samples
                    # for t in range(10):
                    #     X_train_words.append(self.seq_corpus[i])
                    #     X_train_out.append(random.randint(0, len(self.seq_corpus[i])-1))
                    #     X_train_in.append(random.randint(0, len(self.seq_corpus[i])-1))
                    #     X_train_pos.append(self.seq_pos[i])
                    #     Y_train.append(0)
                
                neg_tup = [(m,n) for m in range(self.seq_len[i]) for n in range(self.seq_len[i])]
                for e in neg_tup:
                    #Real Samples
                    if (e[0], e[1]) in real_tup:
                        continue
                    X_train_words.append(self.seq_corpus[i])
                    X_train_out.append(e[0])
                    X_train_in.append(e[1])
                    X_train_pos.append(self.seq_pos[i])
                    Y_train.append(0)

            return X_train_words, X_train_out, X_train_in, X_train_pos, Y_train

        elif (self.mode == 'label'):
            Y_tmp = []
            for i in range(len(self.seq_corpus)):
                for e in self.seq_edges[i]:
                    if(e[2] < 0):
                        continue
                    #Real Samples
                    X_train_words.append(self.seq_corpus[i])
                    X_train_out.append(e[0])
                    X_train_in.append(e[1])
                    X_train_pos.append(self.seq_pos[i])
                    Y_tmp.append(e[2])

            Y_train = np_utils.to_categorical(Y_tmp, num_classes=self.num_labels)

            return X_train_words, X_train_out, X_train_in, X_train_pos, Y_train
            

    def split_valid(self, X_1, X_2, X_3, X_4, Y, v_size=0.1, rand=True):
        if rand == True:
            randomize = np.arange(len(Y))
            np.random.shuffle(randomize)
            X_1, X_2, X_3, X_4, Y = np.array(X_1)[randomize], np.array(X_2)[randomize], np.array(X_3)[randomize], np.array(X_4)[randomize], np.array(Y)[randomize]

        t_size = math.floor(len(Y)*(1-v_size))
        X_1_v = X_1[t_size:]
        X_2_v = X_2[t_size:]
        X_3_v = X_3[t_size:]
        X_4_v = X_4[t_size:]
        Y_v = Y[t_size:]
        X_1 = X_1[:t_size]
        X_2 = X_2[:t_size]
        X_3 = X_3[:t_size]
        X_4 = X_4[:t_size]
        
        Y = Y[:t_size]
        
        return X_1, X_2, X_3, X_4, Y, X_1_v, X_2_v, X_3_v, X_4_v, Y_v
    
    def create_model(self, emb_dim=64, filter_sizes=[2,3,4], drop=0.4, num_filters=[128, 64, 64]):
        sequence_length = self.max_len
        input_words = Input(shape=(sequence_length,))
        input_pos = Input(shape=(sequence_length,))
        input_out = Input(shape=(1,), dtype='int32')
        input_in = Input(shape=(1,), dtype='int32')
        
        pos_vec = Lambda(lambda x: K.one_hot(K.cast(x, 'int32'), len(self.pos_list)))(input_pos)
        pos_vec = Reshape((sequence_length, len(self.pos_list)))(pos_vec)
        out_vec = Lambda(lambda x: K.one_hot(K.cast(x, 'int32'), sequence_length))(input_out)
        out_vec = Reshape((sequence_length, 1))(out_vec)
        in_vec = Lambda(lambda x: K.one_hot(K.cast(x, 'int32'), sequence_length))(input_in)
        in_vec = Reshape((sequence_length, 1))(in_vec)
        #settings
        input_full_dim = self.w2v.wv.syn0.shape[1] + len(self.pos_list) + 2

        word_emb = Embedding(self.w2v.wv.syn0.shape[0]+1,
                self.w2v.wv.syn0.shape[1], #mask_zero=True,
                weights=[np.concatenate([self.w2v.wv.syn0, np.array([np.zeros(self.w2v.wv.syn0.shape[1])])], axis=0)],
                trainable=False)(input_words)
        
        word_emb = Reshape((sequence_length, self.w2v.wv.syn0.shape[1]))(word_emb)
        input_full = Concatenate()([word_emb, pos_vec, out_vec, in_vec])
        input_full = Reshape((sequence_length, input_full_dim))(input_full)
        
        conv_emb = Conv1D(emb_dim, kernel_size=1, padding='valid', kernel_initializer='normal', activation='relu')(input_full)
        conv_emb = Reshape((sequence_length, emb_dim))(conv_emb)
        conv_emb = Conv1D(emb_dim, kernel_size=1, padding='valid', kernel_initializer='normal', activation='relu')(conv_emb)
        conv_emb = Reshape((sequence_length, emb_dim))(conv_emb)
        
        
        conv_0 = Conv1D(num_filters[0], kernel_size=2, padding='valid', kernel_initializer='normal', activation='relu')(conv_emb)
        conv_1 = Conv1D(num_filters[1], kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu')(conv_emb)
        conv_2 = Conv1D(num_filters[2], kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu')(conv_emb)
        
        #Hierachical Convolution
        conv_3 = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu')(conv_0)
        conv_4 = Conv1D(32, kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu')(conv_1)
        conv_5 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu')(conv_2)
        
        #Dilated Convolution
        conv_6 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=3)(conv_0)
        conv_6 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=3)(conv_6)

        conv_7 = Conv1D(32, kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=3)(conv_1)
        conv_7 = Conv1D(32, kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=3)(conv_7)
        
        conv_8 = Conv1D(32, kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=2)(conv_1)
        conv_8 = Conv1D(32, kernel_size=4, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=2)(conv_8)
        
        conv_9 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=2)(conv_1)
        conv_9 = Conv1D(32, kernel_size=3, padding='valid', kernel_initializer='normal', activation='relu', dilation_rate=2)(conv_9)
        


        maxpool_0 = MaxPool1D(pool_size=sequence_length - filter_sizes[0] + 1, strides=None, padding='valid')(conv_0)
        maxpool_1 = MaxPool1D(pool_size=sequence_length - filter_sizes[1] + 1, strides=None, padding='valid')(conv_1)
        maxpool_2 = MaxPool1D(pool_size=sequence_length - filter_sizes[2] + 1, strides=None, padding='valid')(conv_2)
        maxpool_3 = MaxPool1D(pool_size=108, strides=None, padding='valid')(conv_3)
        maxpool_4 = MaxPool1D(pool_size=106, strides=None, padding='valid')(conv_4)
        maxpool_5 = MaxPool1D(pool_size=106, strides=None, padding='valid')(conv_5)
        maxpool_6 = MaxPool1D(pool_size=98, strides=None, padding='valid')(conv_6)
        maxpool_7 = MaxPool1D(pool_size=91, strides=None, padding='valid')(conv_7)
        maxpool_8 = MaxPool1D(pool_size=97, strides=None, padding='valid')(conv_8)
        maxpool_9 = MaxPool1D(pool_size=101, strides=None, padding='valid')(conv_9)

        maxpool_concat = Concatenate(axis=2)([maxpool_0, maxpool_1, maxpool_2, maxpool_3, maxpool_4, maxpool_5, maxpool_6, maxpool_7, maxpool_8, maxpool_9])

        maxpool_flat = Flatten()(maxpool_concat)
        drop_0 = Dropout(drop)(maxpool_flat)
        dense_0 = Dense(128, activation='relu')(drop_0)
        drop_1 = Dropout(drop)(dense_0)
        dense_1 = Dense(64, activation='relu')(drop_1)
        drop_2 = Dropout(drop)(dense_1)
        
        if (self.mode == 'edge'):
            output = Dense(units=1, activation='sigmoid')(drop_2)
        elif (self.mode == 'label'):
            output = Dense(units=self.num_labels, activation='softmax')(drop_2)

        self.model = Model([input_words, input_out, input_in, input_pos], output)

        if (self.mode == 'edge'):
            self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        elif (self.mode == 'label'):
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model.summary()

    def train_model(self, model_filename, epochs=50):
        for e in range(1):
            print('Epoch #'+str(e)+'/'+str(epochs)+'...')
            X_1, X_2, X_3, X_4, Y = self.gen_train_data()
            print(np.array(X_4).shape)
            #print(np.array(X_2).shape)
            #print(np.array(X_3).shape)
            
            X_1, X_2, X_3, X_4, Y, X_1_v, X_2_v, X_3_v, X_4_v, Y_v = self.split_valid(X_1, X_2, X_3, X_4, Y)
            checkpoint = None
            if self.mode == 'label':
                checkpoint = ModelCheckpoint(model_filename, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
                callbacks_list = [checkpoint]
                train_log = self.model.fit(x=[np.array(X_1), np.array(X_2), np.array(X_3), np.array(X_4)], y=np.array(Y), epochs=epochs, batch_size=32, validation_data=([np.array(X_1_v), np.array(X_2_v), np.array(X_3_v), np.array(X_4_v)], np.array(Y_v)), callbacks=callbacks_list, verbose=False)
                print('train_acc = '+str(train_log.history['acc'][-1]))
            
            elif self.mode == 'edge':
                checkpoint = ModelCheckpoint(model_filename, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
                callbacks_list = [checkpoint]
                train_log = self.model.fit(x=[np.array(X_1), np.array(X_2), np.array(X_3), np.array(X_4)], y=np.array(Y), epochs=epochs, batch_size=128, validation_data=([np.array(X_1_v), np.array(X_2_v), np.array(X_3_v), np.array(X_4_v)], np.array(Y_v)), callbacks=callbacks_list, verbose=False)
                print('train_acc = '+str(train_log.history['acc'][-1]))
            
            #self.model.save('last_convdpn_model.h5')
	    
    def load_model(self, model_filename):
        self.model.load_weights(model_filename)

    def test_model(self, s, s_pos):
        test_seq = [self.w2v.wv.vocab[s[i]].index if s[i] in self.w2v.wv else self.unk_idx for i in range(len(s))]
        test_pos = [self.pos_dict[s_pos[i]] for i in range(len(s_pos))]
        test_seq += [0]*(self.max_len-len(s))
        test_pos += [0]*(self.max_len-len(s))
        test_edges = [(i,j) for i in range(len(s)) for j in range(len(s))]
        X_1 = np.array([test_seq for t in test_edges])
        X_2 = np.array([t[0] for t in test_edges])
        X_3 = np.array([t[1] for t in test_edges])
        X_4 = np.array([test_pos for t in test_edges])
        res = self.model.predict([X_1, X_2, X_3, X_4])

        res_graph = []
        if (self.mode == 'label'):
            res_graph_p = []
            for a, b, c in zip(X_2, X_3, res):
                #print(s[a]+'-->'+s[b], self.label_list[np.argmax(c)])
                res_graph.append((a, b, self.label_list[np.argmax(c)]))
                res_graph_p.append((a, b, np.max(c)))
            
            return res_graph, res, res_graph_p
        
        elif (self.mode == 'edge'):
            for a, b, c in zip(X_2, X_3, res):
                #print(s[a]+'-->'+s[b], c[0])
                res_graph.append((a, b, c[0]))

            return res_graph, res

def main():

    label_model = ConvDPN('skip-gram/word2vec.model', mode='label')
    label_model.load_train_data(read_ud_file('UD_data/zh_gsd-ud-train.conllu'))
    label_model.create_model()
    label_model.train_model('convdpn_label_zh_model.h5', epochs=100)

    edge_model = ConvDPN('skip-gram/word2vec.model', mode='edge')
    edge_model.load_train_data(read_ud_file('UD_data/zh_gsd-ud-train.conllu'))
    edge_model.create_model()
    edge_model.train_model('convdpn_edge_zh_model.h5', epochs=20)
    
    
if __name__ == "__main__":
    main()
