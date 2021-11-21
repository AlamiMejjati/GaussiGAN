import argparse
import glob
import numpy as np
from six.moves import range
from dataloader import *
from tensorpack import *
from tensorpack.utils import logger
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorpack.tfutils.export import ModelExporter
from GAN import GANTrainer, GANModelDesc
import os
import sys
import six
from datetime import datetime
import random

seed = 56
# utils.fix_rng_seed(seed)
tf.set_random_seed(seed)

DIS_SCALE = 3
SHAPE = 256
NF = 64  # channel size
N_DIS = 4
N_SAMPLE = 2
N_SCALE = 3
N_RES = 4
STYLE_DIM = 8
STYLE_DIM_z2 = 8
n_upsampling = 5
chs = NF * 8
gauss_std =0.05
LR = 1e-4
enable_argscope_for_module(tf.layers)

def SmartInit(obj, ignore_mismatch=False):
    """
    Create a :class:`SessionInit` to be loaded to a session,
    automatically from any supported objects, with some smart heuristics.
    The object can be:

    + A TF checkpoint
    + A dict of numpy arrays
    + A npz file, to be interpreted as a dict
    + An empty string or None, in which case the sessinit will be a no-op
    + A list of supported objects, to be initialized one by one

    Args:
        obj: a supported object
        ignore_mismatch (bool): ignore failures when the value and the
            variable does not match in their shapes.
            If False, it will throw exception on such errors.
            If True, it will only print a warning.

    Returns:
        SessionInit:
    """
    if not obj:
        return JustCurrentSession()
    if isinstance(obj, list):
        return ChainInit([SmartInit(x, ignore_mismatch=ignore_mismatch) for x in obj])
    if isinstance(obj, six.string_types):
        obj = os.path.expanduser(obj)
        if obj.endswith(".npy") or obj.endswith(".npz"):
            assert tf.gfile.Exists(obj), "File {} does not exist!".format(obj)
            filename = obj
            logger.info("Loading dictionary from {} ...".format(filename))
            if filename.endswith('.npy'):
                obj = np.load(filename, encoding='latin1').item()
            elif filename.endswith('.npz'):
                obj = dict(np.load(filename))
        elif len(tf.gfile.Glob(obj + "*")):
            # Assume to be a TF checkpoint.
            # A TF checkpoint must be a prefix of an actual file.
            return (SaverRestoreRelaxed if ignore_mismatch else SaverRestore)(obj)
        else:
            raise ValueError("Invalid argument to SmartInit: " + obj)

    if isinstance(obj, dict):
        return DictRestore(obj)
    raise ValueError("Invalid argument to SmartInit: " + type(obj))

def tpad(x, pad, mode='CONSTANT',  name=None):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode=mode)

def INLReLU(x, name='IN'):
    with tf.variable_scope(name):
        x = InstanceNorm('in', x)
        x = tf.nn.leaky_relu(x)
    return x

def kl_loss(mean, logvar):
    # shape : [batch_size, channel]
    loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.exp(logvar) - 1 - logvar)
    # loss = tf.reduce_mean(loss)
    return loss

#taken from https://github.com/tomasjakab/imm/
def get_coord(x, other_axis, axis_size):
    # get "x-y" coordinates:
    g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
    g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
    coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
    coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
    g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
    return g_c, g_c_prob

def get_gaussian_maps_2d(mu, sigma, shape_hw, mode='rot'):
  """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """
  with tf.name_scope(None, 'gauss_map', [mu]):

    y = tf.cast(tf.linspace(-1.0, 1.0, shape_hw[0]), tf.float64)
    x = tf.cast(tf.linspace(-1.0, 1.0, shape_hw[1]), tf.float64)

    [x,y] = tf.meshgrid(x,y)
    xy = tf.stack([x, y], axis=-1)
    xy = tf.stack([xy] * nb_landmarks, axis=0)
    xy = tf.reshape(xy, [1, nb_landmarks, shape_hw[0], shape_hw[1], 2])
    mu = tf.reshape(mu, [-1, nb_landmarks, 1, 1, 2])
    # xy = xy[tf.newaxis, :, :, :, :]
    # mu = mu[:,:,tf.newaxis, tf.newaxis,:]
    invsigma = tf.linalg.inv(sigma)
    # invsigma = tf.cast(invsigma, tf.float32)
    invsigma = tf.reshape(invsigma, [-1, nb_landmarks, 1, 2, 2])
    pp = tf.tile(invsigma, [1, 1, shape_hw[1], 1, 1])
    X = xy-mu
    dist = tf.matmul(X,pp)
    dist = tf.reduce_sum((dist*X), axis=-1)

    g_yx = tf.exp(-dist)

    g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])

  return g_yx
# taken from https://github.com/tomasjakab/imm/
def get_gaussian_maps(mu, sigmax, sigmay, covs, shape_hw, mode='rot'):
  """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """
  with tf.name_scope(None, 'gauss_map', [mu]):
    # mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

    x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))
    [x,y] = tf.meshgrid(x,y)
    xy = tf.stack([x, y], axis=-1)
    xy = tf.stack([xy] * nb_landmarks, axis=0)
    xy = xy[None, : ,:, :, :]
  if mode in ['rot', 'flat']:
    mu = mu[:,:,None, None,:]
    # y = tf.reshape(y, [1, 1, shape_hw[0], 1])
    # x = tf.reshape(x, [1, 1, 1, shape_hw[1]])
    # xy = tf.cast(x+y, tf.float32)

    invsigma = tf.stack([sigmay**2, -covs, -covs, sigmax**2], axis=-1)
    invsigma = tf.reshape(invsigma, [-1, nb_landmarks, 2,2])
    denominator = (sigmax*sigmay)**2 - covs**2
    denominator = tf.expand_dims(tf.expand_dims(denominator, -1), -1)
    invsigma = invsigma/(denominator+1e-7)
    invsigma = tf.cast(invsigma, tf.float32)
    pp = tf.tile(invsigma[:, :, None, :, :], [1, 1, shape_hw[1], 1, 1])
    X = xy-mu
    # Xt = tf.transpose(X, [0,1,3,2])
    dist = tf.matmul(X,pp)
    dist = tf.reduce_sum((dist*X), axis=-1)
    # dist = 0.5*tf.matmul(tf.matmul(X, invsigma), Xt)
    # g_y = tf.square(y - mu_y)
    # g_x = tf.square(x - mu_x)
    # dist = (g_y + g_x) * inv_std**2

    if mode == 'rot':
      g_yx = tf.exp(-dist)
    else:
      g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

  else:
    raise ValueError('Unknown mode: ' + str(mode))

  g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
  return g_yx
# get "maximally" different random colors:
#  ref: https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor = 0.5):
  return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
  return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
  max_distance = None
  best_color = None
  for i in range(0,100):
    color = get_random_color(pastel_factor = pastel_factor)
    if not existing_colors:
      return color
    best_distance = min([color_distance(color,c) for c in existing_colors])
    if not max_distance or best_distance > max_distance:
      max_distance = best_distance
      best_color = color
  return best_color

def get_n_colors(n, pastel_factor=0.9):
  colors = []
  for i in range(n):
    colors.append(generate_new_color(colors,pastel_factor = 0.9))
  return colors

def colorize_landmark_maps(maps):
  """
  Given BxHxWxN maps of landmarks, returns an aggregated landmark map
  in which each landmark is colored randomly. BxHxWxN
  """
  n_maps = maps.shape.as_list()[-1]
  # get n colors:
  # colors = get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [tf.expand_dims(maps[..., i], axis=3) * np.reshape(COLORS[i], [1, 1, 1, 3])
           for i in range(n_maps)]
  return tf.reduce_max(hmaps, axis=0)

def get_landmarks(mus, sigma, rotx, roty, rotz, tx, ty, tz, focal=1.):
    assert mus is not None
    assert sigma is not None
    count = 4
    rotXval = rotx
    rotYval = roty
    rotZval = rotz
    rotX = (rotXval) * np.pi / 180
    rotY = (rotYval) * np.pi / 180
    rotZ = (rotZval) * np.pi / 180
    zr = tf.zeros_like(rotY)
    ons = tf.ones_like(rotY)

    RX = tf.stack([tf.stack([ons, zr, zr], axis=-1), tf.stack([zr, tf.cos(rotX), -tf.sin(rotX)], axis=-1),
                   tf.stack([zr, tf.sin(rotX), tf.cos(rotX)], axis=-1)], axis=-1)
    RY = tf.stack([tf.stack([tf.cos(rotY), zr, tf.sin(rotY)], axis=-1), tf.stack([zr, ons, zr], axis=-1),
                   tf.stack([-tf.sin(rotY), zr, tf.cos(rotY)], axis=-1)], axis=-1)
    RZ = tf.stack([tf.stack([tf.cos(rotZ), -tf.sin(rotZ), zr], axis=-1),
                   tf.stack([tf.sin(rotZ), tf.cos(rotZ), zr], axis=-1),
                   tf.stack([zr, zr, ons], axis=-1)], axis=-1)

    # Composed rotation matrix with (RX,RY,RZ)
    R = tf.matmul(tf.matmul(RX, RY), RZ)
    # R = tf.stack([R] * nb_landmarks, axis=0)[None, :, :, :]

    transvec = tf.constant(np.array([[tx, ty, tz]]), dtype=tf.float64)
    transvec = tf.stack([transvec] * nb_landmarks, axis=1)
    transvec = tf.reshape(transvec, [-1, nb_landmarks, 1, 3])
    # transvec = transvec[:, :, tf.newaxis, :]


    px = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    py = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    fvs = tf.ones_like(px) * focal
    zv = tf.zeros_like(px)
    ov = tf.ones_like(px)
    K = tf.stack([tf.stack([fvs, zv, zv], axis=-1), tf.stack([zv, fvs, zv], axis=-1),
                  tf.stack([px, py, ov], axis=-1)], axis=-1)
    K = tf.cast(K, tf.float64)
    K = tf.identity(K, name='K')

    R = tf.cast(R, tf.float64) * tf.ones_like(sigma)
    sigma = tf.linalg.matmul(tf.linalg.matmul(R, sigma), R, transpose_b=True)
    invsigma = tf.linalg.inv(sigma)
    mus = tf.cast(mus, tf.float64)
    mus = tf.transpose(tf.linalg.matmul(R, tf.transpose(mus, [0, 1, 3, 2])), [0, 1, 3, 2]) + transvec

    M0 = tf.matmul(invsigma, tf.matmul(mus, mus, transpose_a=True))
    M0 = tf.matmul(M0, invsigma, transpose_b=True)
    M1 = (tf.matmul(tf.matmul(mus, invsigma), mus, transpose_b=True) - 1)
    M1 = M1 * invsigma

    M = M0 - M1

    Mtmp = tf.constant(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), dtype=tf.float64)
    Mtmp = tf.reshape(Mtmp, [1, 1, 3, 3])
    M = -M + 2 * M * Mtmp
    M33 = tf.gather(tf.gather(M, [0, 1], axis=2), [0, 1], axis=3)
    K33 = tf.gather(tf.gather(K, [0, 1], axis=2), [0, 1], axis=3)
    M31 = tf.gather(tf.gather(M, [0, 1], axis=2), [1, 2], axis=3)
    M23 = tf.gather(tf.gather(M, [0, 2], axis=2), [0, 1], axis=3)
    det_m31 = tf.linalg.det(M31)
    det_m23 = tf.linalg.det(M23)
    det_m33 = tf.linalg.det(M33)
    det_m = tf.linalg.det(M)

    kmul = tf.stack([det_m31, -det_m23], axis=-1)
    kmul = tf.reshape(kmul, [-1, nb_landmarks, 2, 1])
    mup0 = tf.squeeze(tf.matmul(K33, kmul), axis=-1) / tf.reshape(det_m33, [-1,nb_landmarks, 1])
    mup1 = tf.stack([K[:, :, 0, 2], K[:, :, 1, 2]], axis=-1)
    mup = mup0 + mup1

    sigma_w = det_m / det_m33
    # sigma_w = sigma_w[:, :, None, None]
    sigma_w = tf.reshape(sigma_w, [-1, nb_landmarks, 1, 1])
    invm33 = tf.linalg.inv(M33)
    sigmap = -sigma_w * invm33

    gauss_xy_list = []

    mup = tf.identity(mup, name='mu2d')
    sigmap = tf.identity(sigmap, name='sigma2d')
    for i in range(count):
        sz = 2 ** (8 - count + i + 1)
        tmp = get_gaussian_maps_2d(mup, sigmap, [sz, sz])
        tmp = tf.cast(tmp, tf.float32)
        gauss_xy_list.append(tmp)

    mup = tf.cast(mup, tf.float32)
    sigmap = tf.cast(sigmap, tf.float32)
    return gauss_xy_list

def z_sample(mean, logvar):
    eps = tf.random_normal(tf.shape(mean), mean=0.0, stddev=1.0, dtype=tf.float32)

    return mean + tf.exp(logvar * 0.5) * eps

class CheckNumerics(Callback):
    """
    When triggered, check variables in the graph for NaN and Inf.
    Raise exceptions if such an error is found.
    """
    def _setup_graph(self):
        vars = tf.trainable_variables()
        ops = [tf.check_numerics(v, "CheckNumerics['{}']".format(v.op.name)).op for v in vars]
        self._check_op = tf.group(*ops)

    def _before_run(self, _):
        self._check_op.run()

class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 3), 'input'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 1), 'template'),
                tf.placeholder(tf.float32, (None, SHAPE, SHAPE, 1), 'mask'),
                tf.placeholder(tf.float32, (None, 1, 1, 4), 'bbx'),
                tf.placeholder(tf.float32, (None, 1, 1, STYLE_DIM), 'z1'),
                tf.placeholder(tf.float32, (None, 1, 1, STYLE_DIM_z2), 'z2'),
               ]

    @auto_reuse_variable_scope
    def MLP(self, style, channel, nb_upsampling,  name='MLP'):
        with tf.variable_scope(name), \
                argscope([tf.layers.dense],
                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):

            style = tf.layers.flatten(style)
            x = tf.layers.dense(style, channel*2, activation=tf.nn.leaky_relu, name='linear_0')
            x = tf.layers.dense(x, channel*2, activation=tf.nn.leaky_relu, name='linear_1')
            x = tf.layers.dense(x, channel * 2, activation=tf.nn.leaky_relu, name='linear_2')

            x = tf.tile(x[:, np.newaxis], [1, nb_upsampling, 1])

        return x

    @auto_reuse_variable_scope
    def get_musigma(self, nb_blocks=4):
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            x0 = tf.get_variable('cann_lmks', shape=[1, NF*4], dtype=tf.float32, trainable=True,
                                initializer = tf.ones_initializer)
            x = tf.layers.dense(x0, NF * 4, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)

            mus = tf.layers.dense(x, nb_landmarks*3, activation=tf.nn.tanh)
            mus = tf.reshape(mus, [-1, nb_landmarks, 1, 3])
            # mus = mus[:, :, None, :]


            v0 = tf.layers.dense(x, nb_landmarks*3, activation=tf.nn.tanh)
            v0 = tf.reshape(v0, [-1, nb_landmarks, 3])
            v1 = tf.layers.dense(x, nb_landmarks*3, activation=tf.nn.tanh)
            v1 = tf.reshape(v1, [-1, nb_landmarks, 3])

            v0 = tf.math.l2_normalize(v0, axis=-1)
            v1 = tf.linalg.cross(v0, v1)
            v1 = tf.math.l2_normalize(v1, axis=-1)
            v2 = tf.linalg.cross(v1, v0)
            v0 = tf.reshape(v0, [-1, nb_landmarks, 3, 1])
            v1 = tf.reshape(v1, [-1, nb_landmarks, 3, 1])
            v2 = tf.reshape(v2, [-1, nb_landmarks, 3, 1])

            V = tf.concat([v0, v2, v1], axis=-1)
            u = tf.layers.dense(x, nb_landmarks * 3, activation=tf.nn.sigmoid)*0.5+1e-2
            u = tf.reshape(u, [-1,nb_landmarks, 3])
            U = tf.linalg.diag(u)
            sigma = tf.linalg.matmul(tf.linalg.matmul(V,U),V, transpose_b=True)
            sigma = tf.cast(sigma, tf.float64)

        return mus, sigma

    @auto_reuse_variable_scope
    def transform_mu_sigma(self, x, cannmu, cannsigma, nb_blocks=4):
        assert x is not None
        with tf.variable_scope('rotscale'):
            cannsigma = tf.cast(cannsigma, tf.float64)
            x = tf.squeeze(x, axis=[1,2])
            for k in range(nb_blocks):
                x = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)

            x0 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x0 = tf.layers.dense(x0, NF * 4, activation=tf.nn.leaky_relu)

            x1 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x1 = tf.layers.dense(x1, NF * 4, activation=tf.nn.leaky_relu)

            x2 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x2 = tf.layers.dense(x2, NF * 4, activation=tf.nn.leaky_relu)

            x3 = tf.layers.dense(x, NF * 4, activation=tf.nn.leaky_relu)
            x3 = tf.layers.dense(x3, NF * 4, activation=tf.nn.leaky_relu)

            mu_t = tf.layers.dense(x0, nb_landmarks*3, activation=tf.nn.tanh)*0.1
            mu_t = tf.reshape(mu_t, [-1, nb_landmarks, 1, 3])
            mus = cannmu*tf.ones_like(mu_t) + mu_t

            S,U,V = tf.linalg.svd(cannsigma, full_matrices=True)
            S = tf.linalg.diag(tf.sqrt(S))
            Sscale = tf.layers.dense(x1, nb_landmarks*3, activation=tf.nn.tanh)*0.05 + 1.
            Sscale = tf.reshape(Sscale, [-1, nb_landmarks, 3])
            # sfty = tf.linalg.diag(tf.ones_like(Sscale)*0.01)
            Sscale = tf.linalg.diag(Sscale)
            Sscale = tf.cast(Sscale, tf.float64)
            Sscale = S * Sscale #+ sfty

            sigrotx = tf.layers.dense(x2, nb_landmarks, activation=tf.nn.tanh)*np.pi/10.
            sigroty = tf.layers.dense(x2, nb_landmarks, activation=tf.nn.tanh)*np.pi/10.
            sigrotz = tf.layers.dense(x2, nb_landmarks, activation=tf.nn.tanh)*np.pi/10.
            zr = tf.zeros_like(sigrotz)
            os = tf.ones_like(sigrotz)
            RX = tf.stack([tf.stack([os, zr, zr], axis=-1), tf.stack([zr, tf.cos(sigrotx), -tf.sin(sigrotx)], axis=-1),
                           tf.stack([zr, tf.sin(sigrotx), tf.cos(sigrotx)], axis=-1)], axis=-1)
            RY = tf.stack([tf.stack([tf.cos(sigroty), zr, tf.sin(sigroty)], axis=-1), tf.stack([zr, os, zr], axis=-1),
                           tf.stack([-tf.sin(sigroty), zr, tf.cos(sigroty)], axis=-1)], axis=-1)
            RZ = tf.stack([tf.stack([tf.cos(sigrotz), -tf.sin(sigrotz), zr], axis=-1),
                           tf.stack([tf.sin(sigrotz), tf.cos(sigrotz), zr], axis=-1),
                           tf.stack([zr, zr, os], axis=-1)], axis=-1)

            # Composed rotation matrix with (RX,RY,RZ)
            R = tf.matmul(tf.matmul(RX, RY), RZ)
            R = tf.cast(R, tf.float64)
            newU = tf.matmul(R, U*tf.ones_like(R))
            sigmas = tf.matmul(tf.matmul(newU, tf.matmul(Sscale, Sscale)), newU, transpose_b=True)

            theta = tf.layers.dense(x3, 1, activation=tf.nn.tanh)*180.

        return mus, sigmas, theta

    @auto_reuse_variable_scope
    def get_z(self, img):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            x = Conv2D('conv0_0', img, NF, 7, strides=2, activation=INLReLU)
            x = Conv2D('conv0_1', x, NF, 3, strides=1, activation=INLReLU)

            x = Conv2D('conv1_0', x, NF*2, 3, strides=2, activation=INLReLU)
            x = Conv2D('conv1_1', x, NF * 2, 3, strides=1, activation=INLReLU)

            x = Conv2D('conv2_0', x, NF*2, 3, strides=2, activation=INLReLU)
            x = Conv2D('conv2_1', x, NF * 2, 3, strides=1, activation=INLReLU)

            x = Conv2D('conv3_0', x, NF*4, 3, strides=2, activation=INLReLU)
            x = Conv2D('conv3_1', x, NF * 4, 3, strides=1, activation=INLReLU)
            #
            x = Conv2D('conv4_0', x, NF*4, 3, strides=2, activation=INLReLU)
            x = Conv2D('conv4_1', x, NF * 4, 3, strides=1, activation=INLReLU)

            x = Conv2D('conv5_0', x, NF*8, 3, strides=2, activation=INLReLU)
            x = Conv2D('conv5_1', x, NF * 8, 3, strides=1, activation=INLReLU)
            x = tf.nn.max_pool(x, [1, 4,4, 1], strides=[1,1,1,1], padding='VALID')
            x = tf.reshape(x, [-1, NF*8])
            mean = tf.layers.dense(x, STYLE_DIM_z2, name='fcmean')
            mean = tf.reshape(mean, [-1, 1, 1, STYLE_DIM_z2])
            var = tf.layers.dense(x, STYLE_DIM_z2, name='fcvar')
            var = tf.reshape(var, [-1, 1, 1, STYLE_DIM_z2])

        return mean, var

    @auto_reuse_variable_scope
    def get_zm(self, img):
        assert img is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            x = Conv2D('conv0_0', img, NF, 7, strides=2, activation=INLReLU)

            x = Conv2D('conv1_0', x, NF*2, 2, strides=2, activation=INLReLU)

            x = Conv2D('conv2_0', x, NF*4, 2, strides=2, activation=INLReLU)

            x = Conv2D('conv3_0', x, NF*4, 2, strides=2, activation=INLReLU)

            x = Conv2D('conv4_0', x, NF*8, 2, strides=2, activation=INLReLU)
            x = tf.nn.max_pool(x, [1, 4,4, 1], strides=[1,2,2,1], padding='VALID')
            sz = x.shape.as_list()
            x = tf.reshape(x, [-1, sz[1]* sz[2]* sz[3]])
            mean = tf.layers.dense(x, STYLE_DIM_z2, name='fcmean')
            mean = tf.reshape(mean, [-1, 1, 1, STYLE_DIM_z2])
            var = tf.layers.dense(x, STYLE_DIM_z2, name='fcvar')
            var = tf.reshape(var, [-1, 1, 1, STYLE_DIM_z2])

        return mean, var

    @auto_reuse_variable_scope
    def generator(self, pe, chan=3):
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):

            szs = pe[0].shape.as_list()
            in0 = pe[0]
            x = Conv2DTranspose('deconv0_0', in0, NF * 4, 3, strides=1)
            x = Conv2DTranspose('deconv0_1', x, NF * 4, 3, strides=2)
            in1 = tf.concat([x, pe[1]], axis=-1)
            x = Conv2DTranspose('deconv1_0', in1, NF * 2, 3, strides=1)
            x = Conv2DTranspose('deconv1_1', x, NF * 2, 3, strides=2)
            in2 = tf.concat([x, pe[2]], axis=-1)
            x = Conv2DTranspose('deconv2_0', in2, NF, 3, strides=1)
            x = Conv2DTranspose('deconv2_1', x, NF, 3, strides=2)
            in3 = tf.concat([x, pe[3]], axis=-1)
            x = tf.pad(in3, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='SYMMETRIC')
            x = Conv2D('convlast', x, chan, 7, padding='VALID', activation=tf.tanh)
        return x

    @auto_reuse_variable_scope
    def sr_net(self, m, chan=1):
        assert m is not None
        with argscope([Conv2D, Conv2DTranspose], activation=INLReLU):
            m = tf.keras.layers.UpSampling2D(2, data_format=None)(m)
            l = (LinearWrap(m)
                 .Conv2D('conv0_sr', NF, 7, padding='SAME')
                 .Conv2D('conv1_sr', NF, 3, padding='SAME')
                 .Conv2D('conv2_sr', chan, 7, padding='SAME', activation=tf.tanh, use_bias=True)())
        return l

    @staticmethod
    def build_adain_res_block(x, musigma, name, chan, first=False):
        with tf.variable_scope(name), \
            argscope([Conv2D], kernel_size=3, strides=1):

            musigma = tf.reshape(musigma, [-1, 2, chan])
            mu = musigma[:, 0]
            mu = tf.reshape(mu, [-1, 1, 1, chan])
            sigma = musigma[:, 1]
            sigma = tf.reshape(sigma, [-1, 1, 1, chan])

            input = x
            x = Conv2D('conv0', x, chan, 3, activation=tf.nn.leaky_relu, strides=1)
            x = tf.add(mu * InstanceNorm('in_0', x, use_affine=False), sigma, name='adain_0')

            x = Conv2D('conv1', x, chan, 3, activation=tf.nn.leaky_relu, strides=1)
            x = tf.add(mu * InstanceNorm('in_1', x, use_affine=False), sigma, name='adain_1')

            return x+input

    @staticmethod
    def apply_noise(x, name):
        with tf.variable_scope(name):
            sp = x.shape.as_list()
            noise = tf.random_normal([tf.shape(x)[0], sp[1], sp[2], 1], mean=0, stddev=0.5)
            noise = tf.concat([noise]*sp[3], axis=-1)
            gamma = tf.get_variable('gamma', [sp[3]], trainable=True)
            gamma = tf.reshape(gamma, [1,1,1,sp[3]])
        return x+gamma*noise

    @staticmethod
    def build_res_block(x, name, chan):
        with tf.variable_scope(name), \
            argscope([Conv2D], kernel_size=3, strides=1):

            input = x
            x = Conv2D('conv0', x, chan, 3, activation=INLReLU, strides=1)
            x = Conv2D('conv1', x, chan, 3, activation=INLReLU, strides=1)

            return x+input


    @auto_reuse_variable_scope
    def z_reconstructer(self, musigma, dimz ,name='z_reconstructer'):
        with tf.variable_scope(name), \
             argscope([tf.layers.dense],
                      kernel_initializer=tf.contrib.layers.variance_scaling_initializer()):
            musigma = tf.layers.flatten(musigma)
            x = tf.layers.dense(musigma, dimz, activation=tf.nn.leaky_relu, name='linear_0')
            x = tf.layers.dense(x, dimz, activation=tf.nn.leaky_relu, name='linear_1')
            x=tf.reshape(x, [-1,1,1,dimz])
        return x

    @auto_reuse_variable_scope
    def discrim_enc(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=3, strides=2):
            l1 = Conv2D('conv0', img, NF, 7, strides=1, activation=tf.nn.leaky_relu)
            l2 = Conv2D('conv1', l1, NF * 2)
            l3 = Conv2D('conv2', l2, NF * 4)
            features = Conv2D('conv3', l3, NF * 8)
        return features, [l1, l2, l3, features]

    @auto_reuse_variable_scope
    def discrim_classify(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=3, strides=2):
            l1 = Conv2D('conv3', img, NF * 8, strides=2)
            l2 = tf.reduce_mean(l1, axis=[1,2])
            l3 = tf.layers.dense(l2, 1, activation=tf.identity, name='imisreal')
            return l3, [l1]

    @auto_reuse_variable_scope
    def discrim_patch_classify(self, img):
        with argscope(Conv2D, activation=INLReLU, kernel_size=3, strides=2):
            l1 = Conv2D('conv3', img, NF * 8, strides=2)
            l2 = Conv2D('conv4', l1, 1, strides=1, activation=tf.identity, use_bias=True)
            return l2, [l1]

    @auto_reuse_variable_scope
    def style_encoder(self, x):
        chan = NF
        with tf.variable_scope('senc'), argscope([tf.layers.conv2d, Conv2D]):

            x = tpad(x, pad=1, mode='reflect')
            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_0')

            for i in range(3):
                x = tpad(x, pad=1, mode='reflect')
                x = tf.layers.conv2d(x, chan*2,  kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i+1))
                chan*=2

            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i + 2))
            x = tf.layers.conv2d(x, chan, kernel_size=3, strides=2, activation=INLReLU, name='conv_%d' % (i + 3))

            x = tf.layers.flatten(x)
            mean = tf.layers.dense(x, STYLE_DIM, name='fcmean')
            mean = tf.reshape(mean, [-1, 1, 1, STYLE_DIM])
        return mean

    def get_feature_match_loss(self, feats_real, feats_fake, name):
        losses = []

        for i, (real, fake) in enumerate(zip(feats_real, feats_fake)):
            with tf.variable_scope(name):
                fm_loss_real = tf.get_variable('fm_real_%d' % i,
                        real.shape[1:],
                        dtype=tf.float32,
                        trainable=False)

                ema_real_op = moving_averages.assign_moving_average(fm_loss_real,
                    tf.reduce_mean(real, 0), 0.99, zero_debias=False,
                    name='EMA_fm_real_%d' % i)

            loss = tf.reduce_mean(tf.squared_difference(
                fm_loss_real,
                tf.reduce_mean(fake, 0)),
                name='mse_feat_' + real.op.name)

            losses.append(loss)

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_real_op)

        ret = tf.add_n(losses, name='feature_match_loss')
        return ret

    def build_graph(self, img, box, mask, bbx, z, z2):

        with tf.name_scope('preprocess'):
            mask = (mask / 127.5 - 1.0)
            randomrot = tf.random.uniform(shape=[tf.shape(mask)[0], 1], minval=0., maxval=angle*1.)


        def vizN(name, a, b):
            with tf.name_scope(name):
                m = tf.concat(a, axis=2)
                m = tf.image.grayscale_to_rgb(m)
                m = (m + 1.0) * 127.5
                m = tf.clip_by_value(m, 0, 255)

                gm = tf.concat(b, axis=2)*255
                m = tf.concat([m] + [gm], axis=2)
                show = tf.cast(m, tf.uint8, name='viz')
            tf.summary.image(name, show, max_outputs=50)

        # use the initializers from torch
        with argscope([Conv2D, Conv2DTranspose, tf.layers.conv2d]):
            #Let us encode the images

            with tf.variable_scope('mappingnet'):
                mucann, sigmacann = self.get_musigma()
                mucann = tf.identity(mucann, name='mucann')
                sigmacann = tf.identity(sigmacann, name='sigmacann')
            with tf.variable_scope('gen'):
                with tf.variable_scope('senc'):
                    zgt, _ = self.get_z(mask)
                with tf.variable_scope('genlandmarks'):
                    mus, sigmas, thets = self.transform_mu_sigma(zgt, mucann, sigmacann)
                    mucann = mucann * tf.ones_like(mus)
                    sigmacann = sigmacann * tf.ones_like(sigmas)
                    mus = tf.identity(mus, name='mu3d')
                    sigmas = tf.identity(sigmas, name='sigma3d')
                    thets = tf.identity(thets, name='theta3d')
                    zrs = tf.zeros_like(thets)
                    pose_embeddings = get_landmarks(mus, sigmas, zrs, thets, zrs, 0, 0, -2.)
                    pose_embeddings_rot = get_landmarks(mus, sigmas, zrs, thets + randomrot, zrs, 0, 0, -2.)
                    pose_embeddings_cann = get_landmarks(mucann, sigmacann, zrs, zrs, zrs, 0, 0, -2.)
                with tf.variable_scope('genmask'):
                    gen_mask = self.generator(pose_embeddings, 1)
                    gen_mask_rot = self.generator(pose_embeddings_rot, 1)

                ##cycle
                with tf.variable_scope('senc'):
                    zgt_cyc, _ = self.get_z(gen_mask_rot)
                with tf.variable_scope('genlandmarks'):
                    mus_cyc, sigmas_cyc, thets_cyc = self.transform_mu_sigma(zgt_cyc, mucann, sigmacann)
                    pose_embeddings_cyc = get_landmarks(mus_cyc, sigmas_cyc, zrs, thets_cyc - randomrot, zrs, 0, 0, -2.)

            #The final discriminator that takes them both
            discrim_out_mask = []
            discrim_fm_real_mask = []
            discrim_fm_fake_mask = []
            with tf.variable_scope('discrim'):
                with tf.variable_scope('discrim_mask'):
                    def downsample(img):
                        return tf.layers.average_pooling2d(img, 3, 2)

                    D_input_real = mask
                    D_input_fake = gen_mask
                    D_input_fake_rot = gen_mask_rot
                    D_inputs = [D_input_real, D_input_fake, D_input_fake_rot]

                    for s in range(DIS_SCALE):
                        with tf.variable_scope('s%d'%s):
                            if s != 0:
                                D_inputs = [downsample(im) for im in D_inputs]

                            mask_s, mask_recon_s, mask_recon_s_rot = D_inputs


                            with tf.variable_scope('Ax'):
                                Ax_feats_real, Ax_fm_real = self.discrim_enc(mask_s)
                                Ax_feats_fake, Ax_fm_fake = self.discrim_enc(mask_recon_s)
                                Ax_feats_fake_rot, Ax_fm_fake_rot = self.discrim_enc(mask_recon_s_rot)

                            with tf.variable_scope('Ah'):
                                Ah_dis_real, Ah_fm_real = self.discrim_patch_classify(Ax_feats_real)
                                Ah_dis_fake, Ah_fm_fake = self.discrim_patch_classify(Ax_feats_fake)
                                Ah_dis_fake_rot, Ah_fm_fake_rot = self.discrim_patch_classify(Ax_feats_fake_rot)

                            discrim_out_mask.append((Ah_dis_real, Ah_dis_fake))
                            discrim_out_mask.append((Ah_dis_real, Ah_dis_fake_rot))

                            discrim_fm_real_mask += Ax_fm_real + Ah_fm_real + Ax_fm_real + Ah_fm_real
                            discrim_fm_fake_mask += Ax_fm_fake + Ah_fm_fake + Ax_fm_fake_rot + Ah_fm_fake_rot


            vizN('A_recon', [mask, gen_mask, gen_mask_rot], [colorize_landmark_maps(pose_embeddings[-1]),
                                                                   colorize_landmark_maps(pose_embeddings_rot[-1]),
                                                                   colorize_landmark_maps(pose_embeddings_cann[-1]),
                                                                   colorize_landmark_maps(pose_embeddings_cyc[-1])])
        def LSGAN_hinge_loss(real, fake):
            d_real = tf.reduce_mean(-tf.minimum(0., tf.subtract(real, 1.)), name='d_real')
            d_fake = tf.reduce_mean(-tf.minimum(0., tf.add(-fake,-1.)), name='d_fake')
            d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

            g_loss = tf.reduce_mean(-fake, name='g_loss')
            # add_moving_summary(g_loss)
            return g_loss, d_loss

        with tf.name_scope('losses'):
            with tf.name_scope('mask_losses'):
                with tf.name_scope('GAN_loss'):
                    # gan loss
                    G_loss_mask, D_loss_mask = zip(*[LSGAN_hinge_loss(real, fake) for real, fake in discrim_out_mask])
                    G_loss_mask = tf.add_n(G_loss_mask, name='mask_lsgan_loss')/len(G_loss_mask)
                    D_loss_mask = tf.add_n(D_loss_mask, name='mask_Disc_loss')
                with tf.name_scope('FM_loss'):
                    FM_loss_mask = self.get_feature_match_loss(discrim_fm_real_mask, discrim_fm_fake_mask, 'mask_fm_loss')
                with tf.name_scope('m_recon_loss'):
                    mask_recon_loss = tf.reduce_mean(tf.abs(mask - gen_mask), name='mask_recon_loss')
                with tf.name_scope('m_recon_loss_bis'):
                    m_b = tf.reduce_sum(pose_embeddings[-1], axis=-1, keepdims=True)
                    m_r = tf.reduce_sum(pose_embeddings_rot[-1], axis=-1, keepdims=True)
                    mask_recon_loss_bis = tf.reduce_mean(tf.abs((mask+1)*0.5 - m_b), name='mask_recon_loss_bis')
                    mask_recon_loss_rot_bis = tf.reduce_mean(tf.abs((gen_mask_rot + 1) * 0.5 - m_r),
                                                             name='mask_recon_loss_rot_bis')
                with tf.name_scope('gm_cycle_loss'):
                    gm_cycle_loss = tf.reduce_mean(tf.abs(pose_embeddings[-1] - pose_embeddings_cyc[-1]),
                                                   name='cycle_loss')


        self.g_loss = G_loss_mask + LAMBDA * FM_loss_mask + LAMBDA_m * mask_recon_loss \
                      + LAMBDA_gm * mask_recon_loss_bis + LAMBDA_gm_rot * mask_recon_loss_rot_bis + \
                      LAMBDA_cyc * gm_cycle_loss
        self.d_loss = D_loss_mask
        self.collect_variables('gen', 'mappingnet', 'discrim')

        add_moving_summary(G_loss_mask, D_loss_mask,  FM_loss_mask, mask_recon_loss, mask_recon_loss_bis,
                           mask_recon_loss_rot_bis, gm_cycle_loss)

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=LR, trainable=False)
        lrm = ratio*lr
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3), tf.train.AdamOptimizer(lrm, beta1=0.5, epsilon=1e-3)
        # return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def export_compact_m_to_l(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['mask'],
        output_names=['gen/genlandmarks/transpose_3'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_m_to_l.pb'))

def export_compact_z_to_l(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['mask'],
        output_names=['gen/genlandmarks/mu3d', 'gen/genlandmarks/sigma3d', 'gen/genlandmarks/theta3d'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_z_to_l.pb'))

def export_compact_z_to_m(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['gen/senc/add', 'gen/senc_m/add', 'bbx'],
        output_names=['gen/genmask/convlast/output'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_z_to_m.pb'))

def export_compact_l_to_m(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['mask', 'gen/genlandmarks/mu2d', 'gen/genlandmarks/sigma2d'],
        output_names=['gen/genmask/convlast/output'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_l_to_m.pb'))

def export_compact_m_to_m(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications. """
    pred_config = PredictConfig(
        session_init=SmartInit(model_path),
        model=Model(),
        input_names=['bbx', 'mask'],
        output_names=['gen/genmask/convlast/output'])
    ModelExporter(pred_config).export_compact(os.path.join(os.path.dirname(model_path), 'frozen_model_m_to_m.pb'))

def get_data_synth(isTrain=True):

    def get_images(dir1, image_path, istrain):

        dir1 = os.path.abspath(dir1)
        image_path = os.path.abspath(image_path)
        files = sorted(glob.glob(os.path.join(dir1, '*.png')))
        np.random.seed(42)
        np.random.shuffle(files)
        lenfiles = len(files)
        files = files[:int(lenfiles*0.9)] if istrain == 'train' else files[int(lenfiles*0.9):]
        df = data_loader(files, image_path, SHAPE, STYLE_DIM, STYLE_DIM_z2, channel=3, shuffle=isTrain)
        return df

    path_type = 'train' if isTrain else 'val'
    path = args.path
    npy_path = os.path.join(args.data, args.dataset)
    df = get_images(npy_path, path, path_type)
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    df = PrefetchDataZMQ(df, 2 if isTrain else 1)
    return df


class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['input', 'template',  'mask', 'bbx', 'z1', 'z2'], ['A_recon/viz'])
    def _before_train(self):
        global args
        self.val_ds = datagetter(isTrain=False)
        self.val_ds.reset_state()

    def _trigger(self):
        idx = 0
        for iA, tA, mA, bA, z1, z2 in self.val_ds:
            vizA = self.pred(iA, tA, mA, bA, z1, z2)
            self.trainer.monitors.put_image('testA-{}'.format(idx), vizA[0])
            idx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True,
        help='name of the class used')
    parser.add_argument(
        '--data', required=True,
        help='directory containing bounding box annotations, should contain train, val folders')
    parser.add_argument(
        '--path',  help='the path that contains the raw rgb images')
    parser.add_argument('--gpu', default='0', help='nb gpus to use, to use four gpus specify 0,1,2,3')
    parser.add_argument('--dataloader', help='which dataloader to use',
                        default='CocoLoader_center_scaled_keep_aspect_ratio_adacrop', type=str)
    parser.add_argument('--getdata', help='which datagetter to use',
                        default='get_data', type=str)
    parser.add_argument('--nb_epochs', help='hyperparameter', default=200, type=int)
    parser.add_argument('--batch', help='hyperparameter', default=3, type=int)
    parser.add_argument('--testbatch', help='hyperparameter', default=8, type=int)
    parser.add_argument('--LAMBDA', help='hyperparameter', default=10.0, type=float)
    parser.add_argument('--LAMBDA_m', help='hyperparameter', default=100.0, type=float)
    parser.add_argument('--LAMBDA_gm', help='hyperparameter', default=100.0, type=float)
    parser.add_argument('--LAMBDA_gm_rot', help='hyperparameter', default=100.0, type=float)
    parser.add_argument('--LAMBDA_cyc', help='hyperparameter', default=10.0, type=float)
    parser.add_argument('--LAMBDA_KLm', help='hyperparameter', default=0.5,  type=float)
    parser.add_argument('--LAMBDA_KLl', help='hyperparameter', default=1., type=float)
    parser.add_argument('--ratio', help='hyperparameter', default=1., type=float)
    parser.add_argument('--angle', help='hyperparameter', default=90, type=int)
    parser.add_argument('--nb_l', help='hyperparameter', default=5, type=int)
    parser.add_argument('--load', help='load model')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    LAMBDA = args.LAMBDA
    LAMBDA_m = args.LAMBDA_m
    LAMBDA_gm_rot = args.LAMBDA_gm_rot
    LAMBDA_gm = args.LAMBDA_gm
    LAMBDA_cyc = args.LAMBDA_cyc
    ratio = args.ratio
    angle = args.angle
    BATCH = args.batch
    TEST_BATCH = args.testbatch
    nb_epochs = args.nb_epochs
    data_loader = getattr(sys.modules[__name__], args.dataloader)
    nb_landmarks = args.nb_l
    COLORS = get_n_colors(nb_landmarks, pastel_factor=0.0)
    nr_tower = max(get_num_gpu(), 1)
    BATCH = BATCH // nr_tower
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__).split('.')[0]
    logdir = os.path.join('train_log', args.dataset, basename,
                          'Ep%d_r%.1f_L%d_Lm%d_Lgm%d_Lgmrot%d_Lcyc%d_ange%d_nbl%d' %(nb_epochs, ratio, LAMBDA,
                                                                                     LAMBDA_m, LAMBDA_gm, LAMBDA_gm_rot,
                                                                                     LAMBDA_cyc, angle, nb_landmarks),
                          datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger.set_logger_dir(logdir)
    from shutil import copyfile
    namefile = os.path.basename(os.path.realpath(__file__))
    copyfile(namefile, os.path.join(logdir, namefile))
    datagetter = getattr(sys.modules[__name__], args.getdata)
    df = datagetter()
    df = PrintData(df)
    data = QueueInput(df)

    GANTrainer(data, Model(), num_gpu=nr_tower).train_with_defaults(
        callbacks=[
            CheckNumerics(),
            PeriodicTrigger(ModelSaver(), every_k_epochs=int(nb_epochs/2)),
            ScheduledHyperParamSetter(
                'learning_rate',
                [(int(nb_epochs/2), LR), (nb_epochs, 0)], interp='linear'),
            PeriodicTrigger(VisualizeTestSet(), every_k_epochs=50),
        ],
        max_epoch=nb_epochs,
        steps_per_epoch=data.size() // nr_tower,
        session_init=SaverRestore(args.load) if args.load else None,
    )

    export_compact_l_to_m(os.path.join(logdir, 'checkpoint'))
    export_compact_z_to_l(os.path.join(logdir, 'checkpoint'))

