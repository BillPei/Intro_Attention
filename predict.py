#encoding=utf-8
import os, sys
import time
import tensorflow as tf
import numpy as np
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]=""

from base import Model
from reader import reader
from VAEQ import SQS

import segs

reader = reader('./data/', False)
s = SQS(reader)



def txtvector(texts):
	d = []
	for line in texts:
		c = map(reader.vocab.get, line.split(' '))
		c = [i for i in c if i!=None]
		d.append(np.bincount(c, minlength = reader.vocab_size))
	return d

def fq_to_dic(fq_path='/home/yanjianfeng/GoLive/update_anses/fq'):
	f = open(fq_path).read().split('\n')
	fq_dic = {}
	for line in f:
		line = line.split('\t')
		if len(line) != 2:
			continue
		qid, content = line
		qid, sid = qid.split('_')
		if qid in fq_dic.keys():
			if sid == '00':
				fq_dic[qid] = [segs.segs(content.decode('utf-8'), True).encode('utf-8')] + fq_dic[qid]
			elif sid == '01':
				continue
			else:
				fq_dic[qid].append(segs.segs(content.decode('utf-8'), True).encode('utf-8'))
		else:
			if sid != '01':
				fq_dic[qid] = [segs.segs(content.decode('utf-8'), True).encode('utf-8')]
	return fq_dic

if os.path.exists('fq_dic.dump'):
	fq_dic = pickle.load(open('fq_dic.dump'))
else:
	fq_dic = fq_to_dic()
	pickle.dump(fq_dic,open('fq_dic.dump', 'w'))
texts, ids = [], []
for k, v in fq_dic.items():
	for z in v:
		texts.append(z)
		ids.append(k)

q = txtvector(texts)
with tf.Session() as sess:
	mu = s.sents_to_vector(sess, q, q)

def nearest(ques_input):
	ques = segs.segs(ques_input.decode('utf-8'), True).encode('utf-8')
	l = txtvector([ques])
	with tf.Session() as sess:
		mui = s.sents_to_vector(sess, l, l)
	simi = np.sum(mui*mu, axis = 1)
	r = simi.argsort()[::-1]
	for i in range(10):
		print ids[r[i]] + '\t' + texts[r[i]] + str(simi[r[i]])


while True:
	query = raw_input('-->>>')
	nearest(query)
