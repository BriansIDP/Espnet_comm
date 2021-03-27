from __future__ import division

import argparse
import logging
import math
import os
from time import time
from copy import deepcopy
import random

import editdistance

import numpy as np
import six
import torch

from espnet.lm.lm_utils import make_lexical_tree


class KBmeeting(object):
    def __init__(self, vocabulary, meetingpath, charlist, bpe=False):
        """Meeting-wise KB in decoder
        """
        self.meetingdict = {}
        self.meetingdict_sym = {}
        self.meetingmask = {}
        self.meetinglextree = {}
        self.chardict = {}
        self.charlist = charlist
        self.bpe = bpe
        for i, char in enumerate(charlist):
            self.chardict[char] = i

        self.maxlen = 0
        self.unkidx = vocabulary.get_idx('<unk>')
        for filename in os.listdir(meetingpath):
            worddict, wordlist = {}, []
            with open(os.path.join(meetingpath, filename)) as fin:
                for word in fin:
                    word = tuple(word.split()) if bpe else word.strip()
                    worddict[word] = len(wordlist) + 1
                    wordlist.append(word)
            self.meetingdict[filename] = vocabulary.get_ids(wordlist, oov_sym='<blank>')
            self.meetinglextree[filename] = make_lexical_tree(worddict, self.chardict, -1)
            self.maxlen = len(wordlist) if len(wordlist) > self.maxlen else self.maxlen
        # pad meeting wordlist
        for meeting, wordlist in self.meetingdict.items():
            self.meetingdict_sym[meeting] = vocabulary.get_syms(self.meetingdict[meeting])
            self.meetingdict[meeting] = wordlist + [self.unkidx] * (self.maxlen - len(wordlist) + 1)
            self.meetingmask[meeting] = [0] * (len(wordlist)) + [1] * (self.maxlen - len(wordlist)) + [0]
        self.unkidx = self.maxlen
        self.maxlen = self.maxlen + 1
        self.vocab = vocabulary
        self.char_worddict, self.char_dictmask, self.charind, self.char_wordlist = self.get_character_dict()

    def get_meeting_KB(self, meetinglist):
        KBlist = torch.LongTensor([self.meetingdict[meeting] for meeting in meetinglist])
        KBmask = torch.Tensor([self.meetingmask[meeting] for meeting in meetinglist])
        return KBlist, KBmask.byte(), meetinglist

    def get_character_dict(self):
        char_worddict = []
        char_dictmask = []
        char_dictind = []
        char_wordlist = []
        for word in self.vocab.idx2sym:
            if (isinstance(word, tuple) and self.bpe) or not word.startswith('<'):
                char_word = [self.chardict[character] for character in word]
                char_wordlist.append(torch.LongTensor(char_word))
                char_worddict.append(char_word + [0] * (self.vocab.maxwordlen - len(char_word)))
                char_dictmask.append([0] * len(char_word) + [1] * (self.vocab.maxwordlen - len(char_word)))
                char_dictind.append(len(char_word) - 1)
            elif word in self.charlist:
                char_wordlist.append(torch.LongTensor([self.chardict[word]]))
                char_worddict.append([self.chardict[word]] + [0] * (self.vocab.maxwordlen - 1))
                char_dictmask.append([0] + [1] * (self.vocab.maxwordlen - 1))
                char_dictind.append(0)
            else:
                char_wordlist.append(torch.LongTensor([len(self.charlist)]))
                char_worddict.append([len(self.charlist)] + [0] * (self.vocab.maxwordlen - 1))
                char_dictmask.append([0] + [1] * (self.vocab.maxwordlen - 1))
                char_dictind.append(0)
        return torch.LongTensor(char_worddict), torch.LongTensor(char_dictmask).byte(), torch.LongTensor(char_dictind), char_wordlist


class Vocabulary(object):
    def __init__(self, dictfile, bpe=False):
        self.sym2idx = {}
        self.idx2sym = []
        self.maxwordlen = 0
        self.bpe = bpe
        with open(dictfile) as fin:
            for i, line in enumerate(fin):
                if bpe:
                    word = tuple(line.split())
                    self.sym2idx[word] = i
                    self.idx2sym.append(word)
                else:
                    word, ind = line.split()
                    self.sym2idx[word] = int(ind)
                    self.idx2sym.append(word)
                if len(word) > self.maxwordlen:
                    self.maxwordlen = len(word)
        if '<eos>' not in self.sym2idx:
            self.sym2idx['<eos>'] = len(self.idx2sym)
            self.idx2sym.append('<eos>')
        if '<unk>' not in self.sym2idx:
            self.sym2idx['<unk>'] = len(self.idx2sym)
            self.idx2sym.append('<unk>')
        if '<blank>' not in self.sym2idx:
            self.sym2idx['<blank>'] = len(self.idx2sym)
            self.idx2sym.append('<blank>')
        self.ntokens = len(self.idx2sym)

    def get_ids(self, textlist, oov_sym='<unk>'):
        return_list = []
        for word in textlist:
            return_list.append(self.get_idx(word, oov_sym=oov_sym))
        return return_list

    def get_idx(self, word, oov_sym='<unk>'):
        if word not in self.sym2idx:
            return self.sym2idx[oov_sym]
        else:
            return self.sym2idx[word]

    def get_syms(self, ids):
        return [self.idx2sym[i] for i in ids]
