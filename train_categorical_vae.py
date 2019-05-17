"""
Trains a VAE with categorical approximate posterior and prior.
"""

import os
import math
import struct
import shutil
import pickle

import numpy      as np
import scipy      as sp
import tensorflow as tf

from argparse import ArgumentParser
from PIL import Image

random_seed = 0xdeadbeef

# Data
img_shape  = [28, 28]
data_size  = 60000
valid_frac = 0.15

data_path        = 'data'
mnist_train_path = os.path.join(data_path, 'train-images-idx3-ubyte')
mnist_test_path  = os.path.join(data_path, 't10k-images-idx3-ubyte')

# Logging
output_path   = 'output'
log_path      = os.path.join(output_path, 'stats.pyc')
samp_path     = os.path.join(output_path, 'samples')
rec_path      = os.path.join(output_path, 'reconstructions')
samp_grid_dim = 8
rec_count     = 16

# Optimization
iters_per_epoch      = 5000
epoch_count          = 10
batch_size           = 128
step_size            = 3e-4
start_temp, end_temp = 1.0, 0.5
kl_anneal_iters      = 10000
temp_anneal_iters    = 25000

# Model
hidden_size             = 64
latent_code_count       = 20
latent_classes_per_code = 10

def parse_args():
	ap = ArgumentParser()

	"""
	Gumbel-Softmax works well for the reparameterization gradient, but using it for the prior
	and approximate posterior makes optimization drastically more difficult. It's probably best
	to stick with `one_hot`.
	"""
	ap.add_argument('--latent_rep', type=str, default='one_hot', choices=['one_hot', 'relax'])
	ap.add_argument('--grad_est', type=str, default='relax', choices=['relax', 'st'])

	"""
	The temperature needs to be annealed in order to get reasonably coherent samples. Otherwise,
	there is a large gap between the sampling procedure used during training and the one used
	during offline sampling and evaluation. There is a tradeoff between decreasing the minimum
	temperature and increasing the value of the KL term at convergence as a result of increased
	difficulty of optimization.
	"""
	ap.add_argument('--temp_sched', type=str, default='anneal', choices=['fix', 'anneal', 'learn'])
	return ap.parse_args()

def make_wb(n_in, n_out):
	tf.get_variable('w', initializer=tf.random_normal([n_in, n_out], stddev=1 / np.sqrt(n_in)))
	tf.get_variable('b', initializer=tf.zeros([n_out]))

def affine(x):
	w, b = [tf.get_variable(p) for p in 'wb']
	return x @ w + tf.reshape(b, [1, -1])

class Model:
	def parameters(self):
		return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

	def trainable_parameters(self):
		return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Encoder(Model):
	def __init__(self, n_in, n_out, name):
		self.name = name

		with tf.variable_scope(self.name, reuse=False):
			with tf.variable_scope('layer_1', reuse=False): make_wb(n_in, 512)
			with tf.variable_scope('layer_2', reuse=False): make_wb(512, 256)
			with tf.variable_scope('layer_3', reuse=False): make_wb(256, n_out)

	def __call__(self, x):
		with tf.variable_scope(self.name, reuse=True):
			with tf.variable_scope('layer_1', reuse=True): x = tf.nn.relu(affine(x))
			with tf.variable_scope('layer_2', reuse=True): x = tf.nn.relu(affine(x))
			with tf.variable_scope('layer_3', reuse=True): return affine(x)

class Decoder(Model):
	def __init__(self, n_in, n_out, name):
		self.name = name

		with tf.variable_scope(self.name, reuse=False):
			with tf.variable_scope('layer_1', reuse=False): make_wb(n_in, 256)
			with tf.variable_scope('layer_2', reuse=False): make_wb(256, 512)
			with tf.variable_scope('layer_3', reuse=False): make_wb(512, n_out)

	def __call__(self, x):
		with tf.variable_scope(self.name, reuse=True):
			with tf.variable_scope('layer_1', reuse=True): x = tf.nn.relu(affine(x))
			with tf.variable_scope('layer_2', reuse=True): x = tf.nn.relu(affine(x))
			with tf.variable_scope('layer_3', reuse=True): return affine(x)

class TrainDataset:
	def __init__(self, data, batch_size):
		self.data, self.batch_size = data, batch_size
		self.batch_count = data.shape[0] // batch_size
		self.initialize()

	def initialize(self):
		self.cur_index = 0
		self.offsets = np.random.permutation(self.data.shape[0])[:self.batch_size * self.batch_count]

	def next(self):
		if self.cur_index + self.batch_size > self.data.shape[0]:
			self.initialize()

		x = self.data[self.offsets[self.cur_index : self.cur_index + self.batch_size]]
		self.cur_index += self.batch_size
		return x

class TestDataset:
	def __init__(self, data, batch_size):
		self.data, self.batch_size = data, batch_size
		self.batch_count = math.ceil(data.shape[0] / batch_size)
		self.cur_index = 0

	def next(self):
		if self.cur_index >= self.data.shape[0]:
			raise StopIteration()

		n = min(self.batch_size, self.data.shape[0] - self.cur_index)
		x = self.data[self.cur_index : self.cur_index + n]
		self.cur_index += n
		return x

def load_data():
	f = open(mnist_train_path, 'rb')
	assert struct.unpack('>I', f.read(4)) == (2051,)
	assert struct.unpack('>I', f.read(4)) == (data_size,)
	assert struct.unpack('>I', f.read(4)) == (img_shape[0],)
	assert struct.unpack('>I', f.read(4)) == (img_shape[1],)

	d = f.read(img_shape[0] * img_shape[1] * data_size)
	d = np.frombuffer(d, dtype=np.uint8)
	d = np.reshape(d, [data_size, *img_shape])

	train_size = round((1 - valid_frac) * data_size)
	return d[:train_size], d[train_size:]

def assert_non_negative(x):
	with tf.control_dependencies([tf.assert_non_negative(x)]):
		return tf.identity(x)

def random_gumbel(shape, eps=1e-12):
	# It's also important to truncate the upper value because of the outer log.
	u = tf.random_uniform(shape, minval=eps, maxval=1 - eps)
	return -tf.log(-tf.log(u))

def gumbel_softmax_sample(logits, temp):
	g = random_gumbel(tf.shape(logits))
	return tf.nn.softmax((logits + g) / temp)

def gumbel_softmax_log_likelihood(x, logits, temp, eps=1e-12):
	xs = x.get_shape().as_list()
	assert len(xs) == 3
	assert logits.get_shape().as_list() == xs

	"""
	k  = xs[2]
	pi = tf.nn.softmax(logits)
	c  = sp.special.loggamma(k) + (k - 1) * tf.log(temp)
	t1 = -k * tf.log(tf.reduce_sum(pi / (tf.pow(x, temp) + eps), axis=2))
	t2 = tf.reduce_sum(tf.log(pi + eps) - (temp + 1) * tf.log(x + eps), axis=2)
	return c + t1 + t2
	"""

	# This implementation might be more stable, but it still results in NaNs in the gradient for
	# low temperatures like 0.1, so the gradients need to be filtered.
	p = tf.contrib.distributions.RelaxedOneHotCategorical(temp, logits)
	return p.log_prob(x)

def make_train_op(enc, dec, temp, args):
	# MNIST is black-on-white; 0 is white background and 255 is black foreground.
	x_ph = tf.placeholder(tf.uint8, [batch_size, img_size])

	# Dynamic binarization.
	x = tf.random_uniform(tf.shape(x_ph)) < tf.cast(x_ph, tf.float32) / 255
	x = tf.cast(x, tf.float32)

	zs       = [batch_size, latent_code_count, latent_classes_per_code]
	z_logits = tf.reshape(enc(x), zs)
	# XXX: this check fails when `gumbel_softmax_log_likelihood` is used.
	#z_logits = tf.check_numerics(z_logits, 'logits')
	z_relax  = gumbel_softmax_sample(z_logits, temp)

	"""
	Using argmax for the straight-through estimator allows the model to "cheat" in order to make
	the KL loss small: it can make the logits almost uniform. This makes the value of the KL
	training loss uninformative about sample quality. We can fix this by sampling from the
	categorical distribution specified by the logits instead.
	"""
	#z_st = tf.one_hot(tf.argmax(z_logits, axis=2), depth=latent_classes_per_code)

	q    = tf.contrib.distributions.OneHotCategorical(logits=z_logits)
	z_st = tf.cast(q.sample(), tf.float32)

	assert z_relax.get_shape().as_list() == zs
	assert z_st.get_shape().as_list() == zs
	z = z_relax if args.grad_est == 'relax' else tf.stop_gradient(z_st - z_relax) + z_relax

	x_logits = dec(tf.reshape(z, [batch_size, -1]))
	logprobs = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
	assert logprobs.get_shape().as_list() == [batch_size, img_size]
	llp = tf.reduce_sum(logprobs, axis=1)

	if args.latent_rep == 'one_hot':
		probs = tf.nn.softmax(z_logits, axis=2)
		kl_qp = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=probs, logits=z_logits)
		assert kl_qp.get_shape().as_list() == [batch_size, latent_code_count]
		kl_qp = kl_qp + np.log(latent_classes_per_code)
		kl_qp = assert_non_negative(kl_qp)
	else:
		"""
		It's not obvious if a closed-form expression for the KL-divergence between two
		Gumbel-Softmax distributions exists. We use a Monte-Carlo estimate instead.
		"""
		ll = gumbel_softmax_log_likelihood
		prior_logits = tf.ones_like(z_logits) / latent_classes_per_code
		kl_qp = ll(z_relax, z_logits, temp) - ll(z_relax, prior_logits, temp)
		assert kl_qp.get_shape().as_list() == [batch_size, latent_code_count]

	kl_weight = tf.constant(1., dtype=tf.float32)
	kl_weight = tf.get_variable('kl_weight', initializer=kl_weight, trainable=False)
	kl_qp     = tf.reduce_sum(kl_qp, axis=1)
	elb       = llp - kl_weight * kl_qp
	loss      = tf.reduce_mean(-elb)

	var_list = enc.trainable_parameters() + dec.trainable_parameters()
	if args.temp_sched == 'learn': var_list.append(temp)

	opt = tf.train.AdamOptimizer(learning_rate=step_size)

	gv_list   = opt.compute_gradients(loss, var_list=var_list)
	filter    = lambda x: tf.where(tf.is_finite(x), x, tf.zeros_like(x))
	gv_list   = [(filter(g), v) for g, v in gv_list]
	update_op = opt.apply_gradients(gv_list)

	stats = {'elb': -loss, 'll': tf.reduce_mean(llp), 'kl': tf.reduce_mean(kl_qp),
		'kl_weight': kl_weight, 'temp': temp}
	return x_ph, kl_weight, update_op, stats

def make_sample_op(dec):
	"""
	Rather than sampling from the decoder, we set the pixel intensities to the bernoulli
	probabilities. Hence, the outputs will still be somewhat blurry.
	"""

	ind_shape = [samp_grid_dim ** 2, latent_code_count]
	ind = np.random.randint(low=0, high=latent_classes_per_code, size=ind_shape)

	a = np.eye(latent_classes_per_code)
	z = np.reshape(a[np.reshape(ind, [-1])], [*ind_shape, latent_classes_per_code])
	z = tf.constant(z.astype(np.float32))
	x_logits = dec(tf.reshape(z, [samp_grid_dim ** 2, -1]))

	x = tf.nn.sigmoid(x_logits)
	x = tf.cast(tf.rint(255 * x), tf.uint8)
	x = tf.reshape(x, [samp_grid_dim, samp_grid_dim, *img_shape])
	x = tf.transpose(x, [0, 2, 1, 3])
	return tf.reshape(x, [samp_grid_dim * img_shape[0], samp_grid_dim * img_shape[1]])

def make_reconstruction_op(enc, dec, valid_data):
	offsets  = np.random.permutation(valid_data.shape[0])[:rec_count]
	x        = tf.constant(valid_data[offsets])
	z_logits = enc(tf.reshape(tf.cast(x, tf.float32), [rec_count, -1]))
	z_logits = tf.reshape(z_logits, [rec_count, latent_code_count, latent_classes_per_code])
	z_st     = tf.one_hot(tf.argmax(z_logits, axis=2), depth=latent_classes_per_code)
	x_logits = dec(tf.reshape(z_st, [rec_count, -1]))

	x_rec = tf.nn.sigmoid(x_logits)
	x_rec = tf.cast(tf.rint(255 * x_rec), tf.uint8)

	x     = tf.reshape(x, [rec_count, 1, *img_shape])
	x_rec = tf.reshape(x_rec, [rec_count, 1, *img_shape])
	comp  = tf.concat([x, x_rec], axis=1)
	comp  = tf.reshape(comp, [rec_count, 2, img_shape[0], img_shape[1]])
	comp  = tf.transpose(comp, [0, 2, 1, 3])
	return tf.reshape(comp, [rec_count * img_shape[0], 2 * img_shape[1]])

def linear_ramp(cur, start, end, total):
	return start + (end - start) * min(cur / total, 1)

def make_eval_op(enc, dec):
	# MNIST is black-on-white; 0 is white background and 255 is black foreground.
	x_ph = tf.placeholder(tf.uint8, [batch_size, img_size])
	x    = tf.cast(tf.rint(x_ph / 255), tf.float32)

	zs       = [batch_size, latent_code_count, latent_classes_per_code]
	z_logits = tf.reshape(enc(x), zs)

	"""
	For the ELB we actually want to minimize on the validation set, both the approximate
	posterior and the prior are given by categorical distributions.
	"""
	q = tf.contrib.distributions.OneHotCategorical(logits=z_logits)
	z = tf.cast(q.sample(), tf.float32)

	x_logits = dec(tf.reshape(z, [batch_size, -1]))
	logprobs = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_logits)
	assert logprobs.get_shape().as_list() == [batch_size, img_size]
	llp = tf.reduce_sum(logprobs, axis=1)

	probs = tf.nn.softmax(z_logits, axis=2)
	kl_qp = -tf.nn.softmax_cross_entropy_with_logits_v2(labels=probs, logits=z_logits)
	assert kl_qp.get_shape().as_list() == [batch_size, latent_code_count]
	kl_qp = kl_qp + np.log(latent_classes_per_code)
	kl_qp = assert_non_negative(kl_qp)

	kl_qp = tf.reduce_sum(kl_qp, axis=1)
	elb   = llp - kl_qp
	return x_ph, elb

if __name__ == '__main__':
	args = parse_args()
	train_data, valid_data = load_data()
	if os.path.exists(output_path): shutil.rmtree(output_path)
	for path in [samp_path, rec_path]: os.makedirs(path)

	np.random.seed(seed=random_seed)
	tf.random.set_random_seed(seed=random_seed)

	img_size = img_shape[0] * img_shape[1]
	code_size = latent_code_count * latent_classes_per_code
	enc = Encoder(n_in=img_size, n_out=code_size, name='encoder')
	dec = Decoder(n_in=code_size, n_out=img_size, name='decoder')

	temp = tf.constant(1., dtype=tf.float32)

	if args.temp_sched == 'anneal':
		temp = tf.get_variable('temp', initializer=temp, trainable=False)
	elif args.temp_sched == 'learn':
		temp = tf.get_variable('temp', initializer=temp, trainable=True)

	x_ph, kl_weight, update_op, stats = make_train_op(enc, dec, temp, args)
	samp_op = make_sample_op(dec)
	rec_op = make_reconstruction_op(enc, dec, valid_data)
	x_ph_eval, elb_eval = make_eval_op(enc, dec)

	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	train_iter = TrainDataset(train_data, batch_size)
	train_stats, valid_elbs = [], {}

	for cur_epoch in range(epoch_count):
		print(f"On epoch {cur_epoch + 1}.")

		for cur_iter in range(iters_per_epoch):
			abs_iter = cur_epoch * iters_per_epoch + cur_iter
			kl_weight.load(linear_ramp(abs_iter, 0, 1, kl_anneal_iters), sess)

			if args.temp_sched == 'anneal':
				cur_temp = linear_ramp(abs_iter, start_temp, end_temp, temp_anneal_iters)
				temp.load(cur_temp, sess)

			x_next = np.reshape(train_iter.next(), [-1, img_size])
			_, cur_stats = sess.run([update_op, stats], feed_dict={x_ph: x_next})
			train_stats.append(cur_stats)

			if cur_iter == 0 or (cur_iter + 1) % 100 == 0:
				print(f"* Iteration {cur_iter + 1}: {cur_stats}")

		print(f"Validating model.")
		elb_list = []
		valid_iter = TestDataset(valid_data, batch_size)

		for cur_iter in range(valid_iter.batch_count):
			x_next = np.reshape(valid_iter.next(), [-1, img_size])
			pad = batch_size - x_next.shape[0]
			assert pad >= 0

			if pad != 0:
				z = np.zeros([pad, img_size], dtype=x_next.dtype)
				x_next = np.concatenate([x_next, z], axis=0)

			elb = sess.run(elb_eval, feed_dict={x_ph_eval: x_next})
			elb_list.append(elb[:-pad])

		elb_list = np.concatenate(elb_list, axis=0)
		elb = np.mean(elb_list)
		valid_elbs[(cur_epoch + 1) * iters_per_epoch] = elb
		print(f"Validation ELB: {elb}")

		print(f"Saving samples.")
		samp = sess.run(samp_op)
		img_path = os.path.join(samp_path, f'samples_epoch_{cur_epoch + 1}.png')
		Image.fromarray(samp).save(img_path, mode='L')

		print(f"Saving reconstructions.")
		rec = sess.run(rec_op)
		img_path = os.path.join(rec_path, f'reconstructions_epoch_{cur_epoch + 1}.png')
		Image.fromarray(rec).save(img_path, mode='L')

		print()

	stat_list = lambda k: np.array([stats[k] for stats in train_stats])
	train_stats_lists = {k : stat_list(k) for k in train_stats[0].keys()}
	info = {'train_stats': train_stats_lists, 'valid_elbs': valid_elbs}
	with open(log_path, 'wb') as f: pickle.dump(info, f)
