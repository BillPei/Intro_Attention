#encoding=utf-8
import os, sys
import time
import tensorflow as tf
import numpy as np
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]=""

from base import Model
from feeder import reader
from intro_attention import SQS


reader = reader('./data/', False)
s = SQS(reader, dropout=False)

def txtvector(texts):
	dv, dl  = [], []
	for line in texts:
		c = map(reader.vocab.get, line.split(' '))
		c = [i for i in c if i!=None]
		if len(c) > 32:
			dv.append(c[:32])
			dl.append([1]*32)
		else:
			dv.append(c + [reader.vocab['_PAD_']]*(32-len(c)))
			dl.append([1]*len(c)+[0]*(32-len(c)))
	return dv, dl


test_data = open('./data/test.pair').read().split('\n')
ques, ans, label = [], [], []
for line in test_data:
	line = line.split('\t')
	if len(line) == 3:
		ques.append(line[0])
		ans.append(line[1])
		label.append(int(line[2]))
qv, ql = txtvector(ques)
av, al = txtvector(ans)

with tf.Session() as sess:
	simi = s.predict(sess, qv, ql, av, al)


label = np.array(label)
def testing(threshold):
	N = len(label)
	T = 0
	for i in range(len(label)):
		if simi[i] > threshold:
			if label[i] == 1:
				T += 1
		else:
			if label[i] == 0:
				T += 1

	return T

best_t = 0
for t in np.arange(0, 1, 0.02):
	T = testing(t)
	if T > best_t:
		best_t = T
		bt = t
print '2000 samples, threshold %s,accuracy %s' %(bt, 1.0*best_t/len(label))
if len(sys.argv) <= 1:
	sys.exit(0)

# ------------------------------------------------------unk
unk = open('data/unkown.txt').read().split('\n')
unkv = txtvector(unk)
with tf.Session() as sess:
	unkv = s.sents_to_vector(sess, unkv)

def testing_unk(threshold):
	T = 0
	for i in range(len(unkv)):
		v = unkv[i]
		#欧斯距离
		#simi = np.sqrt(np.sum(np.square(kv - v),1))
		#点积距离取sigmoid
		simi = 1.0/(1.0+np.exp(-np.mean(kv*v, 1)))
		if max(simi) < threshold:
			T += 1
	return 1.0*T/len(unk)


#-----------------------------------------------------
fq_dic = pickle.load(open('fq_dic.dump'))
kb_text, kb_label = [], []
for k, v in fq_dic.items():
	for vi in v:
		kb_text.append(vi)
		kb_label.append(k)

kb_text_v = txtvector(kb_text)
with tf.Session() as sess:
	kv = s.ans_to_vector(sess, kb_text_v)
ques = open('data/test_kb.pair').read().split('\n')
ques = [i.split('\t') for i in ques if len(i.split('\t')) == 2]
ques, labels =txtvector([i[0] for i in ques]), [i[1] for i in ques]
with tf.Session() as sess:
	qv = s.sents_to_vector(sess, ques)

def nearest(v, kv, label, threshold = bt):
	simi = 1.0/(1.0+np.exp(-np.mean(kv*v, 1)))
	#simi = np.sqrt(np.sum(np.square(kv - v),1))
	r = simi.argsort()[::-1]
	if simi[r[0]] < threshold:
		return -2
	for i in range(5):
		if simi[r[i]] < threshold:
			return -1
		if kb_label[r[i]] == label:
			return i
	return None

for threshold in np.arange(0, 1, 0.05):
	pred = []
	for i in range(len(qv)):
		pred.append(nearest(qv[i], kv, labels[i], threshold))
	pred = np.array(pred)
	N = len(pred)*1.0
	tunk = testing_unk(threshold)
	print '1000 samples, threshold: %s,No return: %s, top 5: %s, 对于不在知识库问题的正确排除率: %s' %(threshold,
		sum(pred == -2)/N,
		sum((pred < 5)*(pred >= 0))/N,
		tunk)



