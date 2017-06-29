#encoding=utf-8
import os, sys
import time
import tensorflow as tf
import numpy as np

from base import Model
from feeder import reader
from cnn_modula import cnnx

class SQS(Model):

	def __init__(self, reader,
		dropout = True,
		embed_dim = 256,
		rnn_dim = 128,
		num_filters = 128,
		learning_rate = 0.01,
		model_dir = 'model_dir'):

		Model.__init__(self, model_dir)

		self.reader = reader
		self.vocab_size = len(self.reader.vocab)
		self.embed_dim = embed_dim
		self.rnn_dim = rnn_dim
		self.num_filters = num_filters
		self.learning_rate = learning_rate
		self.cnnx = cnnx(num_filters)

		self.build_model(dropout)


	def placeholder_initializer(self):
		self.q = tf.placeholder(tf.int32, [None, 32], name = 'q')
		self.ql = tf.placeholder(tf.int32, [None], name = 'ql')
		self.a = tf.placeholder(tf.int32, [None, 32], name = 'a')
		self.al = tf.placeholder(tf.int32, [None], name = 'al')
		self.label = tf.placeholder(tf.int32, [None], name = 'label')
		self.label = tf.to_float(self.label)

	def variables_initializer(self):

		self.w_embedding = tf.get_variable('w_embedding', [self.vocab_size, self.embed_dim],
			initializer = tf.random_uniform_initializer(-1.0, 1.0))

		self.H = tf.get_variable('H', [self.num_filters*3, self.num_filters])
		self.Hb = tf.get_variable('Hb', [self.num_filters])
		self.H2 = tf.get_variable('H2', [self.num_filters, 1])

	def build_model(self, dropout):
		self.placeholder_initializer()
		self.variables_initializer()

		self.qv = tf.nn.embedding_lookup(self.w_embedding, self.q)
		self.av = tf.nn.embedding_lookup(self.w_embedding, self.a)
		#adding droppout to the input data during trainning .
		if dropout:
			self.qv = tf.nn.dropout(self.qv, keep_prob=0.8)
			self.av = tf.nn.dropout(self.av, keep_prob=0.8)
		self.global_step = tf.Variable(0, name = 'global_step', trainable=False)

		with tf.variable_scope('lstm'):
			cell = tf.nn.rnn_cell.LSTMCell(self.rnn_dim,
				forget_bias = 2.0,
				use_peepholes = True,
				state_is_tuple = True)
			cell_r = tf.nn.rnn_cell.LSTMCell(self.rnn_dim,
				forget_bias = 2.0,
				use_peepholes = True,
				state_is_tuple = True)
			rnnqv, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, self.qv, sequence_length = self.ql, dtype = tf.float32)
		with tf.variable_scope('lstm_2'):
			cell = tf.nn.rnn_cell.LSTMCell(self.rnn_dim,
				forget_bias = 2.0,
				use_peepholes = True,
				state_is_tuple = True)
			cell_r = tf.nn.rnn_cell.LSTMCell(self.rnn_dim,
				forget_bias = 2.0,
				use_peepholes = True,
				state_is_tuple = True)
			rnnav, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, self.av, sequence_length = self.al, dtype = tf.float32)

		rnnqv = tf.concat(2, rnnqv)
		rnnav = tf.concat(2, rnnav)

		with tf.variable_scope('Attend'):
			self.wet1 = tf.batch_matmul(tf.nn.l2_normalize(self.qv, -1),
				tf.transpose(tf.nn.l2_normalize(self.av, -1), [0, 2, 1]))
			self.wet2 = tf.batch_matmul(tf.nn.l2_normalize(rnnqv, -1),
				tf.transpose(tf.nn.l2_normalize(rnnav, -1), [0, 2, 1]))

			self.wet = tf.concat(3, [tf.expand_dims(self.wet1, -1),
				tf.expand_dims(self.wet2, -1)])

		self.pooled = self.cnnx.convolution(self.wet)

		with tf.variable_scope('final_accessment'):
			#pred = tf.nn.bias_add(tf.matmul(self.pooled, self.H), self.Hb)
			#pred = tf.nn.tanh(pred)
			#pred = tf.matmul(pred, self.H2)
			_ = tf.get_variable('_', [self.num_filters*2, 1])
			pred = tf.matmul(self.pooled, _)

			pred = tf.squeeze(pred, [1])
			self.simi = tf.sigmoid(pred)
			self.p = tf.reduce_mean(self.simi)
			loss = tf.nn.sigmoid_cross_entropy_with_logits(targets=self.label, logits = pred)
		#loss = -self.label*tf.log(self.pred+1e-12) - (1.0-self.label)*tf.log(1.0-self.pred+1e-12)
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

		for i in range(12400):
			q, ql, a, al, y = reader.iterator()
			aa, ab, ac = self.sess.run([self.train_op, self.loss, self.p],
				feed_dict = {self.q: q, self.ql: ql, self.a: a, self.al: al, self.label:y})
			print 'AFTER %s steps, the loss %s, %s' %(count, ab, ac)

			self.global_step.assign(count).eval()
			if count%100 == 0:
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

	def predict(self, sess, qv, ql, av, al):
		self.sess = sess
		self.sess.run(tf.initialize_all_variables())
		self.load()

		simi = self.sess.run(self.simi, feed_dict={self.q: qv, self.ql: ql, self.a: av, self.al: al})
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
