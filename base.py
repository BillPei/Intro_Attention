import os,sys
import tensorflow as tf

class Model(object):

	def __init__(self, model_dir='model_dir'):
		self.model_dir = model_dir

	def save(self, global_step=None, checkpoint_dir=False):
		self.saver = tf.train.Saver()

		if checkpoint_dir:
			model_dir = checkpoint_dir
		else:
			model_dir = self.model_dir

		if not os.path.exists(model_dir):
			os.makedirs(model_dir)

		if self.sess:
			self.saver.save(self.sess,
				os.path.join(model_dir, 'model.ckpt'))
			print 'successfull save the model to\t' + os.path.join(model_dir, 'model.ckpt')

	def load(self, checkpoint_dir=False):
		self.saver = tf.train.Saver()

		if checkpoint_dir:
			model_dir = checkpoint_dir
		else:
			model_dir = self.model_dir

		ckpt = tf.train.get_checkpoint_state(model_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(model_dir, ckpt_name))
			print 'successfull restore the sesstion\t' + ckpt.model_checkpoint_path