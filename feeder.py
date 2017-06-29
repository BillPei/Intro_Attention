#encoding=utf-8
import os
import itertools
import numpy as np
import pickle
from collections import Counter



class reader(object):
  def __init__(self, data_path, trainning=True):
    self.data_path = data_path
    train_path = os.listdir(data_path)
    vocab_path = os.path.join(data_path, "vocab")


    if os.path.exists(vocab_path):
      self.vocab = pickle.load(open(vocab_path, 'r'))
      if trainning:
        self._read_fq_dic()
        self._file_to_data(train_path)


    else:
      self._build_vocab(train_path, vocab_path)
      self._read_fq_dic()
      self._file_to_data(train_path)

    self.idx2word = {v:k for k, v in self.vocab.items()}
    self.vocab_size = len(self.vocab)
    self.i = 0
    self.i2 = 0


  def _build_vocab(self, file_path, vocab_path):

    data = []
    for dp in file_path:
      if 'pair' in dp:
        print 'creating vocabulary from ' + dp
        d = open(self.data_path+dp).read().split('\n')
        d = [i for j in d for i in j.split('\t')]
        data += d
      elif 'append' in dp:
        print 'creating vocabulary from ' + dp
        d = open(self.data_path+dp).read().split('\n')
        d = [i for j in d for i in j.split('\t')[:1]]
        data += d
    counter = Counter([i for line in data for i in line.split(' ')])

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = [i for i in count_pairs if i[1] >= 5]
    words = [i[0] for i in count_pairs]
    self.vocab = dict(zip(words, range(len(words))))
    self.vocab['_PAD_'] = len(self.vocab)
    print 'the coabsize is\t' + str(len(self.vocab))

    pickle.dump(self.vocab, open(vocab_path, 'w'))

  def _file_to_data(self, file_path):
    self.ques, self.ans, self.label = [], [], []
    for dp in file_path:
      if 'train.pair' in dp:
        print 'transforming the data from ' + self.data_path + dp
        d = open(self.data_path+dp).read().split('\n')
        for line in d:
          try:
            line = line.split('\t')
            if len(line) == 3:
              q, a, y = line
              q = map(self.vocab.get, q.split(' '))
              a = map(self.vocab.get, a.split(' '))
              y = int(y)
              q = [x for x in q if x != None]
              a = [x for x in a if x != None]
              if (len(q) >= 2) and (len(a) >= 2):
                self.ques.append(q)
                self.ans.append(a)
                self.label.append(y)
          except:
            continue
      if 'train.append' in dp:
        print 'transforming the data from' + self.data_path + dp
        d= open(self.data_path+dp).read().split('\n')
        self.qapend, self.aapend = [], []
        for line in d:
          line = line.split('\t')
          if len(line) == 2:
            q, a = line
            q = map(self.vocab.get, q.split(' '))
            q = [x for x in q if x != None]
            if len(q) >=2:
              self.qapend.append(q)
              self.aapend.append(a)

  def _read_fq_dic(self):
    fq_dic  = pickle.load(open(self.data_path+'fq_dic.dump'))
    self.fq_dic= {}
    for k, v in fq_dic.items():
      self.fq_dic[k] = []
      for vi in v:
        vi = map(self.vocab.get, vi.split(' '))
        vi = [x for x in vi if x != None]
        self.fq_dic[k].append(vi)

  def trans(self, data):
    dv, dl = [], []
    for line in data:
      if len(line) > 32:
        dv.append(line[:32])
        dl.append([1]*32)
      else:
        dv.append(line + [self.vocab['_PAD_']]*(32-len(line)))
        dl.append([1]*len(line)+[0]*(32-len(line)))
    return dv, dl

  def iterator1(self, N = 1000):

    if self.i + N >= len(self.ques):
      self.i  = len(self.ques)%N

    x = self.ques[self.i: self.i+N]
    y = self.ans[self.i: self.i+N]
    z = self.label[self.i: self.i+N]

    self.i += N
    qv, ql = self.trans(x)
    av, al = self.trans(y)

    return qv, ql, av, al, z

  def iterator2(self, N = 200):
    if self.i2 + N >= len(self.qapend):
      self.i2 = len(self.qapend)%N

    x, y, z = [], [], []
    for i in range(self.i2, self.i2+N):
      try:
        yi = self.fq_dic[self.aapend[i]][0]
        zi  = 1
      except:
        yi = [self.vocab['_PAD_']]*10
        zi  = 0
      x.append(self.qapend[i])
      y.append(yi)
      z.append(zi)
      for _ in range(5):
        k = np.random.choice(self.fq_dic.keys())
        if k != self.aapend[i]:
          x.append(self.qapend[i])
          y.append(self.fq_dic[k][0])
          z.append(0)

    self.i2 += N
    qv, ql = self.trans(x)
    av, al = self.trans(y)

    return qv, ql, av, al, z

  def iterator(self, N = 1000):
    if np.random.choice([0]) == 1:
      return self.iterator2(N/2)
    else:
      return self.iterator1(N)



