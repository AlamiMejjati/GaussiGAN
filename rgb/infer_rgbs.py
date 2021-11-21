import cv2
from PIL import Image, ImageTk
import io
import tensorflow as tf
from dataloader import mask_gen_hor_cs_kar_beauty_bg
import numpy as np
import os
import glob
from html import HTML
import skvideo.io
import argparse
tf.enable_eager_execution()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', required=True, help='folder where checkpoint is stored')
parser.add_argument('--masks', default= '/home/youssef/Documents/phdYoop/Stamps/dataset/val/synthgiraffe')
parser.add_argument('--ims', default='/home/youssef/Documents/phdYoop/datasets/synthgiraffe/beauty/PNG')
parser.add_argument('--fullrotation', default=0)
args = parser.parse_args()
l_to_m_model = os.path.join(args.modelpath, 'frozen_model_l_to_m.pb')
m_to_l_model = os.path.join(args.modelpath, 'frozen_model_z_to_l.pb')
m_to_t_model = os.path.join(args.modelpath, 'frozen_model_m_to_t.pb')

orig_im = args.masks
mpaths = sorted(glob.glob(os.path.join(orig_im, '*.png')))
np.random.seed(42)
np.random.shuffle(mpaths)
lenm = len(mpaths)
mpaths = mpaths[int(0.9*len(mpaths)):][:20]
impath = args.ims
nb_landmarks = int(l_to_m_model.split('_nbl')[1].split('/')[0])
SHAPE = 256
bgpath = '../datasets/backgrounds/'
bgfiles = sorted(glob.glob(os.path.join(bgpath, '*.png')))
dataloader = mask_gen_hor_cs_kar_beauty_bg(mpaths, impath, bgfiles, SHAPE, 8, 8)

def get_img_data(f, maxsize=(200, 200), first=True):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)

def create_template(nr, r):
    # r = np.reshape(np.squeeze(r) * SHAPE, [10, 2])
    r = np.squeeze(r*SHAPE)
    nr = nr*SHAPE
    margin = 3
    box = np.array([0, 0, 0, 0])
    bbx = np.zeros([1, SHAPE, SHAPE, 1])
    mins = np.min(nr, axis=0)
    maxs = np.max(nr, axis=0)

    box[1] = int(mins[1]) - margin
    box[3] = int(maxs[1]) + margin
    box[0] = int(mins[0]) - margin
    box[2] = int(maxs[0]) + margin

    if box[0] < 0: box[0] = 0
    if box[1] < 0: box[1] = 0
    if box[3] > SHAPE: box[3] = SHAPE - 1
    if box[2] > SHAPE: box[2] = SHAPE - 1

    if box[3] == box[1]:
        box[3] += 1
    if box[0] == box[2]:
        box[2] += 1

    bbx[:, box[0]:box[2], box[1]:box[3], :] = 1

    box = np.reshape(box, [1,1,1,4])/SHAPE

    return bbx, box

#taken from https://github.com/tomasjakab/imm/
def get_coord(x, other_axis, axis_size):
    # get "x-y" coordinates:
    g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
    g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
    coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size))  # W
    coord_pt = tf.reshape(coord_pt, [1, axis_size, 1])
    g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1)
    return g_c, g_c_prob

# taken from https://github.com/tomasjakab/imm/
def get_gaussian_maps(mu, shape_hw, inv_std, mode='ankush'):
  """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """
  with tf.name_scope(None, 'gauss_map', [mu]):
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]

    y = np.linspace(0., 1.0, shape_hw[0])
    x = np.linspace(0., 1.0, shape_hw[1])

  if mode in ['rot', 'flat']:
    mu_y, mu_x = np.expand_dims(mu_y, -1), np.expand_dims(mu_x, -1)

    y = np.reshape(y, [1, 1, shape_hw[0], 1])
    x = np.reshape(x, [1, 1, 1, shape_hw[1]])

    g_y = np.square(y - mu_y)
    g_x = np.square(x - mu_x)
    dist = (g_y + g_x) * inv_std**2

    if mode == 'rot':
      g_yx = np.exp(-dist)
    else:
      g_yx = tf.exp(-tf.pow(dist + 1e-5, 0.25))

  elif mode == 'ankush':
    y = tf.reshape(y, [1, 1, shape_hw[0]])
    x = tf.reshape(x, [1, 1, shape_hw[1]])

    g_y = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_y - y) * inv_std)))
    g_x = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_x - x) * inv_std)))

    g_y = tf.expand_dims(g_y, axis=3)
    g_x = tf.expand_dims(g_x, axis=2)
    g_yx = tf.matmul(g_y, g_x)  # [B, NMAPS, H, W]

  else:
    raise ValueError('Unknown mode: ' + str(mode))

  g_yx = np.transpose(g_yx, [0, 2, 3, 1])
  return g_yx

# get "maximally" different random colors:
#  ref: https://gist.github.com/adewes/5884820
def get_random_color(pastel_factor = 0.5):
  return [(x+pastel_factor)/(1.0+pastel_factor) for x in [np.random.uniform(0,1.0) for i in [1,2,3]]]

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

COLORS = get_n_colors(nb_landmarks, pastel_factor=0.9)
def colorize_landmark_maps(maps):
  """
  Given BxHxWxN maps of landmarks, returns an aggregated landmark map
  in which each landmark is colored randomly. BxHxWxN
  """
  n_maps = maps.shape[-1]
  # get n colors:
  # colors = get_n_colors(n_maps, pastel_factor=0.0)
  hmaps = [np.expand_dims(maps[..., i], axis=3) * np.reshape(COLORS[i], [1, 1, 1, 3])
           for i in range(n_maps)]
  return np.max(hmaps, axis=0)

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
    transvec = transvec[:, :, tf.newaxis, :]


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
    M = -M + 2 * M * Mtmp[tf.newaxis, tf.newaxis, :, :]
    M33 = tf.gather(tf.gather(M, [0, 1], axis=2), [0, 1], axis=3)
    K33 = tf.gather(tf.gather(K, [0, 1], axis=2), [0, 1], axis=3)
    M31 = tf.gather(tf.gather(M, [0, 1], axis=2), [1, 2], axis=3)
    M23 = tf.gather(tf.gather(M, [0, 2], axis=2), [0, 1], axis=3)
    det_m31 = tf.linalg.det(M31)
    det_m23 = tf.linalg.det(M23)
    det_m33 = tf.linalg.det(M33)
    det_m = tf.linalg.det(M)

    mup0 = tf.squeeze(tf.matmul(K33, tf.stack([det_m31, -det_m23], axis=-1)[:, :, :, tf.newaxis]), axis=-1) / (
        det_m33[:, :, tf.newaxis])
    mup1 = tf.stack([K[:, :, 0, 2], K[:, :, 1, 2]], axis=-1)
    mup = mup0 + mup1

    sigma_w = det_m / det_m33
    sigma_w = sigma_w[:, :, None, None]
    invm33 = tf.linalg.inv(M33)
    sigmap = -sigma_w * invm33

    gauss_xy_list = []
    mup = tf.cast(mup, tf.float32)
    sigmap = tf.cast(sigmap, tf.float32)
    mup = tf.identity(mup, name='mu2d')
    sigmap = tf.identity(sigmap, name='sigma2d')
    gm2d = get_gaussian_maps_2d(mup, sigmap, [256, 256])

    return mup, sigmap, gm2d

def get_landmarks_(mus, sigma, rotx, roty, rotz, tx, ty, tz, focal=1.):
    assert mus is not None
    assert sigma is not None
    count = 4
    rotXval = rotx
    rotYval = roty
    rotZval = rotz
    rotX = (rotXval) * np.pi / 180
    rotY = (rotYval) * np.pi / 180
    rotZ = (rotZval) * np.pi / 180
    # Rotation matrices around the X,Y,Z axis
    ons = tf.ones_like(rotY)
    zr = tf.zeros_like(rotY)
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

    transvec = tf.constant(np.array([[tx, ty, tz]]), dtype=tf.float32)
    transvec = tf.stack([transvec] * nb_landmarks, axis=1)
    transvec = transvec[:, :, None, :]


    px = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    py = tf.zeros([tf.shape(mus)[0], nb_landmarks])
    fvs = tf.ones_like(px) * focal
    zv = tf.zeros_like(px)
    ov = tf.ones_like(px)
    K = tf.stack([tf.stack([fvs, zv, zv], axis=-1), tf.stack([zv, fvs, zv], axis=-1),
                  tf.stack([px, py, ov], axis=-1)], axis=-1)
    K = tf.identity(K, name='K')

    R = R * tf.ones_like(sigma)
    sigma = tf.linalg.matmul(tf.linalg.matmul(R, sigma), R, transpose_b=True)
    # mus = tf.linalg.matmul(mus, tf.linalg.inv(R)) + transvec
    mus = tf.transpose(tf.linalg.matmul(R, tf.transpose(mus, [0, 1, 3, 2])), [0, 1, 3, 2]) + transvec
    invsigma = tf.linalg.inv(sigma)

    M0 = tf.matmul(invsigma, tf.matmul(mus, mus, transpose_a=True))
    M0 = tf.matmul(M0, invsigma, transpose_b=True)
    M1 = (tf.matmul(tf.matmul(mus, invsigma), mus, transpose_b=True) - 1)
    M1 = M1 * invsigma

    M = M0 - M1

    Mtmp = tf.constant(np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]]), dtype=tf.float32)
    M = -M + 2 * M * Mtmp[None, None, :, :]
    M33 = tf.gather(tf.gather(M, [0, 1], axis=2), [0, 1], axis=3)
    K33 = tf.gather(tf.gather(K, [0, 1], axis=2), [0, 1], axis=3)
    M31 = tf.gather(tf.gather(M, [0, 1], axis=2), [1, 2], axis=3)
    M23 = tf.gather(tf.gather(M, [0, 2], axis=2), [0, 1], axis=3)
    det_m31 = tf.linalg.det(M31)
    det_m23 = tf.linalg.det(M23)
    det_m33 = tf.linalg.det(M33)
    det_m = tf.linalg.det(M)

    mup0 = tf.squeeze(tf.matmul(K33, tf.stack([det_m31, -det_m23], axis=-1)[:, :, :, None]), axis=-1) / (
        det_m33[:, :, None])
    mup1 = tf.stack([K[:, :, 0, 2], K[:, :, 1, 2]], axis=-1)
    mup = mup0 + mup1

    sigma_w = det_m / det_m33
    sigma_w = sigma_w[:, :, None, None]
    invm33 = tf.linalg.inv(M33)
    sigmap = -sigma_w * invm33

    gm = get_gaussian_maps_2d(mup, sigmap, [256, 256])

    return mup, sigmap, gm

def get_gaussian_maps_2d(mu, sigma, shape_hw, mode='rot'):
  """
  Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
  given the gaussian centers: MU [B, NMAPS, 2] tensor.

  STD: is the fixed standard dev.
  """
  with tf.name_scope(None, 'gauss_map', [mu]):

    y = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[0]))

    x = tf.to_float(tf.linspace(-1.0, 1.0, shape_hw[1]))
    [x,y] = tf.meshgrid(x,y)
    xy = tf.stack([x, y], axis=-1)
    xy = tf.stack([xy] * nb_landmarks, axis=0)
    xy = xy[tf.newaxis, : ,:, :, :]
    mu = mu[:,:,tf.newaxis, tf.newaxis,:]
    invsigma = tf.linalg.inv(sigma)
    invsigma = tf.cast(invsigma, tf.float32)
    pp = tf.tile(invsigma[:, :, tf.newaxis, :, :], [1, 1, shape_hw[1], 1, 1])
    X = xy-mu
    dist = tf.matmul(X,pp)
    dist = tf.reduce_sum((dist*X), axis=-1)

    g_yx = tf.exp(-dist)

    g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])

  return g_yx



with tf.gfile.GFile(m_to_l_model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph_m_to_l:
    tf.import_graph_def(graph_def, name='')

with tf.gfile.GFile(l_to_m_model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph_l_to_m:
    tf.import_graph_def(graph_def, name='')

with tf.gfile.GFile(m_to_t_model, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
with tf.Graph().as_default() as graph_m_to_t:
    tf.import_graph_def(graph_def, name='')
# We access the input and output nodes


mu2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/mu2d:0')
sigma2d_ltm = graph_l_to_m.get_tensor_by_name('gen/genlandmarks/sigma2d:0')
genm = graph_l_to_m.get_tensor_by_name('gen/genmask/convlast/output:0')

mask_mtl = graph_m_to_l.get_tensor_by_name('mask:0')
# im_mtl = graph_m_to_l.get_tensor_by_name('input:0')
# bbx = graph_m_to_l.get_tensor_by_name('bbx:0')
mu3d_mtl = graph_m_to_l.get_tensor_by_name('gen/genlandmarks/mu3d:0')
sigma3d_mtl = graph_m_to_l.get_tensor_by_name('gen/genlandmarks/sigma3d:0')
theta3d_mtl = graph_m_to_l.get_tensor_by_name('gen/genlandmarks/theta3d:0')

mask_mtt = graph_m_to_t.get_tensor_by_name('mask:0')
input_mtt = graph_m_to_t.get_tensor_by_name('input:0')
imgcrop_mtt = graph_m_to_t.get_tensor_by_name('preprocess/imgcrop:0')
if '_ete_' in args.modelpath:
    z_mtt = graph_m_to_t.get_tensor_by_name('gen/genim/sencim/ztexture:0')
    genim_mtt = graph_m_to_t.get_tensor_by_name('gen/genim/imgen/gen_im:0')
    input_mtl = graph_m_to_l.get_tensor_by_name('input:0')
else:
    z_mtt = graph_m_to_t.get_tensor_by_name('genim/sencim/ztexture:0')
    genim_mtt = graph_m_to_t.get_tensor_by_name('genim/imgen/gen_im:0')
mu2d_mtt = graph_m_to_t.get_tensor_by_name('gen/genlandmarks/mu2d:0')
sigma2d_mtt = graph_m_to_t.get_tensor_by_name('gen/genlandmarks/sigma2d:0')


if args.fullrotation:
    angle=360
    eval_dir = os.path.join(os.path.dirname(m_to_l_model), 'eval_360/ims')
    os.makedirs(eval_dir, exist_ok=True)
else:
    angle = int(args.modelpath.split('_ange')[1].split('_')[0])
    eval_dir = os.path.join(os.path.dirname(m_to_l_model), 'eval/ims')
    os.makedirs(eval_dir, exist_ok=True)

html = HTML(os.path.dirname(eval_dir), 'show images')
html.add_header('%s' % ('3d gm perspective'))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
ims = []
texts = []
links = []
mus = []
sigmas = []

nb_frames = 60
nb_zs = 10
nb_inter_zs = 20
angle_range = list(np.linspace(0., angle, nb_frames))
for its, (np_im, template_np, np_m, r, z_np, z2_np) in enumerate(dataloader):
    ims = []
    texts = []
    links = []
    template = (template_np / 127.5 - 1.0)
    zs = [np.random.normal(size=[1, 1, 1, 8]) for zit in range(nb_zs)]
    allzs = []
    iminterpolations=[]
    for k in range(1,nb_zs):
        allzs += [zs[k]*p + zs[k-1]*[1-p] for p in np.linspace(0,1,nb_inter_zs)]
    with tf.Session(graph=graph_m_to_l, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_l:
        if '_ete_' in args.modelpath:
            mu3d, sig3d, thet3d = sess_m_to_l.run([mu3d_mtl, sigma3d_mtl, theta3d_mtl], feed_dict={mask_mtl: np_m,
                                                                                                   input_mtl: np_im})
        else:
            mu3d, sig3d, thet3d = sess_m_to_l.run([mu3d_mtl, sigma3d_mtl, theta3d_mtl], feed_dict={mask_mtl: np_m})
        # mu3d, sig3d, thet3d = sess_m_to_l.run([mu3d_mtl, sigma3d_mtl], feed_dict={mask_mtl: np_m})
        sig3d = sig3d.astype(np.float64)
        sess_m_to_l.close()

    cv2.imwrite(os.path.join(eval_dir, 'im%d.png'%its), np_im.squeeze()[:,:,::-1])
    cv2.imwrite(os.path.join(eval_dir, 'm%d.png' % its), 255 - np_m.squeeze())
    ims.append('./ims/im%d.png'%its)
    texts.append('input im')
    links.append('./ims/im%d.png'%its)
    gmframes = []
    mframes = []
    imframes = []
    fgframes = []
    np.save(os.path.join(eval_dir, 'm_%d.npy'%its), mu3d)
    np.save(os.path.join(eval_dir, 'sig3d_%d.npy'%its), sig3d)
    for x in angle_range:
        x = np.array([[x]], dtype=np.float32)
        zrs = tf.zeros_like(x)
        mu2d, sigma2d, gm2d = get_landmarks(mu3d, sig3d, zrs, thet3d[0,0] + x*1., zrs, 0., 0., -2.)
        with tf.Session(graph=graph_l_to_m, config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_l_to_m:

            m_final = sess_l_to_m.run(genm, feed_dict={mu2d_ltm: mu2d.numpy(),
                                                       sigma2d_ltm: sigma2d.numpy()})
            m_final = (m_final + 1) * 0.5
            img_crop = template * (1 - m_final) - m_final
            m_final = (np.squeeze(m_final* 255)).astype(np.uint8)
            sess_l_to_m.close()

        mframes.append(255 - m_final)
        gmf = (255 -np.squeeze(colorize_landmark_maps(gm2d))*255).astype(np.uint8)
        gmframes.append(gmf[:,:,::-1])

        with tf.Session(graph=graph_m_to_t,
                        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_t:
            im_final = sess_m_to_t.run(genim_mtt, feed_dict={mu2d_mtt: mu2d.numpy(),
                                                             sigma2d_mtt: sigma2d.numpy(),
                                                             z_mtt: z_np,
                                                             imgcrop_mtt: img_crop,
                                                             mask_mtt: m_final[None, :, :, None],
                                                             input_mtt: template_np})
            im_final = (im_final + 1) * 0.5
            im_final = (np.squeeze(im_final * 255)).astype(np.uint8)
            imframes.append(im_final)
            fgframes.append((im_final*(m_final[:,:,None]/255.)).astype(np.uint8))

            sess_m_to_t.close()

        if x == 0:
            gm2drgb = colorize_landmark_maps(gm2d)
            cv2.imwrite(os.path.join(eval_dir, '2dgm_%d.png'%its), (255. - gm2drgb.squeeze()*255).astype(np.uint8))
            ims.append('./ims/2dgm_%d.png'%its)
            texts.append('2dgm')
            links.append('./ims/2dgm_%d.png'%its)

            cv2.imwrite(os.path.join(eval_dir, 'genim_%d.png'%its), im_final[:,:,::-1])
            ims.append('./ims/genim_%d.png'%its)
            texts.append('gen im')
            links.append('./ims/genim_%d.png'%its)

            cv2.imwrite(os.path.join(eval_dir, 'genm_%d.png'%its), 255 - m_final)
            ims.append('./ims/genm_%d.png'%its)
            texts.append('gen m')
            links.append('./ims/genm_%d.png'%its)
            with tf.Session(graph=graph_m_to_t,
                            config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess_m_to_t:
                for zk in allzs:
                    im_final = sess_m_to_t.run(genim_mtt, feed_dict={mu2d_mtt: mu2d.numpy(),
                                                                     sigma2d_mtt: sigma2d.numpy(),
                                                                     z_mtt: zk,
                                                                     imgcrop_mtt: img_crop,
                                                                     mask_mtt: m_final[None, :, :, None],
                                                                     input_mtt: template_np})

                    im_final = (im_final + 1) * 0.5
                    im_final = (np.squeeze(im_final* 255)).astype(np.uint8)
                    iminterpolations.append(im_final)
                sess_m_to_t.close()

    vidgmpath = os.path.join(eval_dir, 'vid_gm%d.mp4' % its)
    vidgmpathavi = os.path.join(eval_dir, 'vid_gm%d.avi' % its)
    skvideo.io.vwrite(vidgmpath, gmframes)
    skvideo.io.vwrite(vidgmpathavi, gmframes)
    ims.append(os.path.join('./ims', os.path.basename(vidgmpath)))
    texts.append('gmvid')
    links.append(os.path.join('./ims', os.path.basename(vidgmpath)))


    vidmpath = os.path.join(eval_dir, 'vid_m%d.mp4' % its)
    vidmpathavi = os.path.join(eval_dir, 'vid_m%d.avi' % its)
    skvideo.io.vwrite(vidmpath, mframes)
    skvideo.io.vwrite(vidmpathavi, mframes)
    ims.append(os.path.join('./ims', os.path.basename(vidmpath)))
    texts.append('mvid')
    links.append(os.path.join('./ims', os.path.basename(vidmpath)))

    vidimpath = os.path.join(eval_dir, 'vid_im%d.mp4' % its)
    vidimpathavi = os.path.join(eval_dir, 'vid_im%d.avi' % its)
    skvideo.io.vwrite(vidimpath, imframes)
    skvideo.io.vwrite(vidimpathavi, imframes)
    ims.append(os.path.join('./ims', os.path.basename(vidimpath)))
    texts.append('imvid')
    links.append(os.path.join('./ims', os.path.basename(vidimpath)))

    vidfgpath = os.path.join(eval_dir, 'vid_fg%d.mp4' % its)
    vidfgpathavi = os.path.join(eval_dir, 'vid_fg%d.avi' % its)
    skvideo.io.vwrite(vidfgpath, fgframes)
    skvideo.io.vwrite(vidfgpathavi, fgframes)

    vidlatentpath = os.path.join(eval_dir, 'vid_latent%d.mp4' % its)
    vidlatentpathavi = os.path.join(eval_dir, 'vid_latent%d.avi' % its)
    skvideo.io.vwrite(vidlatentpath, iminterpolations)
    skvideo.io.vwrite(vidlatentpathavi, iminterpolations)
    ims.append(os.path.join('./ims', os.path.basename(vidlatentpath)))
    texts.append('imlatent')
    links.append(os.path.join('./ims', os.path.basename(vidlatentpath)))

    html.add_im_vid(ims, texts, links, width=256)
    html.save()
