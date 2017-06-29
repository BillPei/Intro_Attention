#encoding=utf-8
import os, sys
import time
import tensorflow as tf
import numpy as np

from base import Model
from feeder import reader

class SQS(Model):

	def __init__(self, reader,
		dropout = True,
		embed_dim = 256,
		h_dim = 128,
		v_dim = 128,
		learning_rate = 0.02,
		model_dir = 'model_dir'):

		Model.__init__(self, model_dir)

		self.reader = reader
		self.vocab_size = len(self.reader.vocab)
		self.embed_dim = embed_dim
		self.h_dim = h_dim
		self.v_dim = v_dim
		self.learning_rate = learning_rate

		self.build_model(dropout)

	def placeholder_initializer(self):
		self.q = tf.placeholder(tf.int32, [None, 32], name = 'q')
		self.ql = tf.placeholder(tf.int32, [None, 32], name = 'ql')
		self.a = tf.placeholder(tf.int32, [None, 32], name = 'a')
		self.al = tf.placeholder(tf.int32, [None, 32], name = 'al')
		self.label = tf.placeholder(tf.int32, [None], name = 'label')
		self.label = tf.to_float(self.label)

		self.ql, self.al = tf.to_float(self.ql), tf.to_float(self.al)

	def variables_initializer(self):

		self.w_embedding = tf.get_variable('w_embedding', [self.vocab_size, self.embed_dim],
			initializer = tf.random_uniform_initializer(-1.0, 1.0))
		self.F = tf.get_variable('attend_F', [self.embed_dim, self.h_dim])
		self.F_b = tf.get_variable('attend_F_b', [self.h_dim])

		self.G = tf.get_variable('compare_G', [self.embed_dim*2, self.v_dim])
		self.G_b = tf.get_variable('G_b', [self.v_dim])

		self.H = tf.get_variable('H', [self.v_dim*2, self.v_dim*2],
			initializer = tf.random_uniform_initializer(-0.01, 0.01))
		self.H_b = tf.get_variable('H_b', [self.v_dim*2])
		self.H_F = tf.get_variable('H_F', [self.v_dim*2, 1])

	def build_model(self, dropout):
		self.placeholder_initializer()
		self.variables_initializer()

		self.qv = tf.nn.embedding_lookup(self.w_embedding, self.q)
		self.qv = self.qv*tf.expand_dims(self.ql, -1)
		self.av = tf.nn.embedding_lookup(self.w_embedding, self.a)
		self.av = self.av*tf.expand_dims(self.al, -1)
		#adding droppout to the input data during trainning .
		if False:
			self.qv = tf.nn.dropout(self.qv, keep_prob=0.6)
			self.av = tf.nn.dropout(self.av, keep_prob=0.6)
		self.global_step = tf.Variable(0, name = 'global_step', trainable=False)

		with tf.variable_scope('Attend'):
			self.qvt = tf.contrib.layers.linear(self.qv, self.h_dim)
			self.avt = tf.contrib.layers.linear(self.av, self.h_dim)

			self.qvt = tf.nn.relu(self.qvt)
			self.avt = tf.nn.relu(self.avt)

			#Mask所有填充的向量为0
			self.qvt = self.qvt*tf.expand_dims(self.ql, -1)
			self.avt = self.avt*tf.expand_dims(self.al, -1)

			self.attwet = tf.batch_matmul(self.qvt, tf.transpose(self.avt, [0, 2, 1]))
			# 计算新的加入attention之后的b矩阵
			self.av_wet = tf.nn.l2_normalize(self.attwet, dim = 2)
			self.av_trans = tf.batch_matmul(self.av_wet, self.av)
			# 计算新的加入attention之后的a矩阵
			self.qv_wet = tf.nn.l2_normalize(self.attwet, dim = 1)
			self.qv_trans = tf.batch_matmul(tf.transpose(self.qv_wet, [0, 2, 1]), self.qv)

		with tf.variable_scope('Compare'):
			self.qav1 = tf.concat(2, [self.qv, self.av_trans])
			self.qav2 = tf.concat(2, [self.av, self.qv_trans])

			# G函数转换
			#self.qav1 = tf.contrib.layers.linear(self.qav1, self.v_dim)
			#self.qav2 = tf.contrib.layers.linear(self.qav2, self.v_dim)

			self.qav1 = tf.nn.relu(self.qav1)
			self.qav2 = tf.nn.relu(self.qav2)

		with tf.variable_scope('Aggregate'):
			self.v1 = tf.reduce_sum(self.qav1, 1)
			self.v2 = tf.reduce_sum(self.qav2, 1)
			#下面是按照原论文的feed forward network+linear layer， 但是效果似乎并不太好；
			#self.v = tf.concat(1, [self.v1, self.v2])
			#self.v_trans = tf.nn.bias_add(tf.matmul(self.v, self.H), self.H_b)
			#self.v_trans = tf.nn.tanh(self.v_trans)
			#self.pred = tf.matmul(self.v_trans, self.H_F)
			#直接v1*M*v2效果要好很多，不知道为什么
			self.v1 = tf.contrib.layers.linear(self.v1, self.embed_dim*2)
			self.pred = tf.reduce_mean(self.v1*self.v2, -1)
			self.pred = tf.reshape(self.pred, [-1])
			self.pred = tf.nn.sigmoid(self.pred)

			self.simi = self.pred
			self.p = tf.reduce_mean(self.pred)

		loss = -self.label*tf.log(self.pred+1e-12) - (1.0-self.label)*tf.log(1.0-self.pred+1e-12)
		self.loss = tf.reduce_mean(loss)
		self.train_op = tf.contrib.layers.optimize_loss(loss = self.loss,
			global_step = tf.contrib.framework.get_global_step(),
			learning_rate = self.learning_rate,
			clip_gradients = 10.0,
			optimizer = "Adam")

	def train(self, sess):
		self.sess = sess
		self.sess.run(tf.initialize_all_variables())
		self.load()
		count = self.global_step.eval()

		for i in range(10000):
			q, ql, a, al, y = reader.iterator()
			aa, ab, ac = self.sess.run([self.train_op, self.loss, self.p],
				feed_dict = {self.q: q, self.ql: ql, self.a: a, self.al: al, self.label:y})
			print 'AFTER %s steps, the loss %s, %s' %(count, ab, ac)

			self.global_step.assign(count).eval()
			if count%300 == 0:
				self.save(global_step=i)
				os.system('python test.py')

			count += 1

	def sents_to_vector(self, sess, q):
		self.sess = sess
		self.sess.run(tf.initialize_all_variables())
		self.load()

		mu = self.sess.run(self.mu, feed_dict={self.q:q})
		return mu

	def ans_to_vector(self, sess, a):
		self.sess = sess
		self.sess.run(tf.initialize_all_variables())
		self.load()

		mu = self.sess.run(self.mux, feed_dict={self.a:a})
		return mu

	def predict(self, sess, q, ql, a, al):
		self.sess = sess
		self.sess.run(tf.initialize_all_variables())
		self.load()

		simi = self.sess.run(self.simi, feed_dict={self.q: q, self.a:a, self.ql:ql, self.al:al})
		return simi




def txtvector(texts):
	d = []
	for line in texts:
		c = map(reader.vocab.get, segs(line.decode('utf-8'), True).encode('utf-8').split(' '))
		c = [i for i in c if i!=None]
		d.append(np.bincount(c, minlength = reader.vocab_size))
	return d

if __name__ == '__main__':
	if sys.argv[1] == 'train':
		reader = reader('./data/')
		s = SQS(reader, dropout=True)
		with tf.Session() as sess:
			s.train(sess)
