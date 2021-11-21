import cv2
from tensorpack import *
import numpy as np
import os



class SynthLoader(RNGDataFlow):
    """ Produce images read from a list of files. """

    def __init__(self, files, main_dir, shape, shapez, shapez2, channel=3, resize=None, shuffle=False, flip=0.5):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(files), "No image files given to ImageFromFile!"
        self.files = files
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.resize = resize
        self.shuffle = shuffle
        self.indexes = list(range(len(self.files)))
        self.main_dir = main_dir
        self.maxsize = shape
        self.shapez = shapez
        self.shapez2 = shapez2
        self.flipprob = flip

    def __len__(self):
        return len(self.files)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indexes)
        for i in self.indexes:
            mf = self.files[i]
            filename = os.path.basename(mf).split('.png')[0] + '_1.png'


            f = os.path.join(self.main_dir, filename)
            im = cv2.imread(f, self.imread_mode)
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            assert im is not None, f
            if self.channel == 3:
                im = im[:, :, ::-1]
            if self.resize is not None:
                im = cv2.resize(im, tuple(self.resize[::-1]))
            if self.channel == 1:
                im = im[:, :, np.newaxis]

            original_size = max(im.shape[0], im.shape[1])
            resized_width = self.maxsize
            resized_height = self.maxsize
            box, crp_dir = self.find_bbx(m)
            im = im[box[0]:box[2], box[1]:box[3], :]
            m = m[box[0]:box[2], box[1]:box[3]]
            whilecond = True
            if self.shuffle:
                im_m = np.concatenate([im, np.expand_dims(m, axis=-1)], axis=-1)
                if np.random.uniform(0, 1) > self.flipprob:
                    im_m = cv2.flip(im_m, 1)

                resized = cv2.resize(im_m, (resized_height, resized_width))
                im = resized[:, :, :-1]
                maskj = resized[:, :, -1]

            else:
                im = cv2.resize(im, (resized_height, resized_width))
                maskj = cv2.resize(m, (resized_height, resized_width))

            maskj = np.expand_dims(maskj, axis=-1)

            box = np.array([0, 0, 0, 0])

            # Compute Bbx coordinates

            margin = 0
            bbx = np.zeros_like(maskj)
            xs = np.nonzero(np.sum(maskj, axis=0))[0]
            ys = np.nonzero(np.sum(maskj, axis=1))[0]
            box[1] = xs.min() - margin
            box[3] = xs.max() + margin
            box[0] = ys.min() - margin
            box[2] = ys.max() + margin

            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[3] > resized_height: box[3] = resized_height - 1
            if box[2] > resized_width: box[2] = resized_width - 1

            if box[3] == box[1]:
                box[3] += 1
            if box[0] == box[2]:
                box[2] += 1

            bbx[box[0]:box[2], box[1]:box[3], :] = 1
            # box[2] = box[2] - box[0]
            # box[3] = box[3] - box[1]
            box = box * 1.
            box[0] = box[0] / resized_width
            box[2] = box[2] / resized_width
            box[1] = box[1] / resized_height
            box[3] = box[3] / resized_height
            box = np.reshape(box, [1, 1, 4])

            z = np.random.normal(size=[1, 1, self.shapez])
            z2 = np.random.normal(size=[1, 1, self.shapez2])
            yield [im, bbx, maskj, box, z, z2]

    def find_bbx(self, maskj):

        maskj = np.expand_dims(maskj, axis=-1)
        box = np.array([0, 0, 0, 0])

        # Compute Bbx coordinates

        margin = 3
        xs = np.nonzero(np.sum(maskj, axis=0))[0]
        ys = np.nonzero(np.sum(maskj, axis=1))[0]
        box[1] = xs.min() - margin
        box[3] = xs.max() + margin
        box[0] = 0
        box[2] = maskj.shape[0]

        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0

        h = box[2] - box[0]
        w = box[3] - box[1]
        if h < w:
            diff = w - h
            half = int(diff / 2)
            box[0] -= half
            if box[0] < 0:
                box[2] -= box[0]
                box[0] = 0
            else:
                box[2] += diff - half

            if box[2] > maskj.shape[0]:
                box[2] = maskj.shape[0]
        else:
            diff = h - w
            half = int(diff / 2)
            box[1] -= half
            if box[1] < 0:
                box[3] -= box[1]
                box[1] = 0
            else:
                box[3] += diff - half
            if box[3] > maskj.shape[1]:
                box[3] = maskj.shape[1]

        if box[3] == box[1]:
            box[3] += 1
        if box[0] == box[2]:
            box[2] += 1

        return box, h-w

class mask_gen_hor_cs_kar_albedo():
    """ Produce images read from a list of files. """

    def __init__(self, m_files, main_dir, shape, shapez, shapez2, channel=3):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(m_files)
        self.m_files = m_files
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.maxsize = shape
        self.shapez = shapez
        self.shapez2 = shapez2
        self.main_dir = main_dir

    def __len__(self):
        return len(self.m_files)

    def __iter__(self):

        for j in self.m_files:

            m = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
            filename = os.path.basename(j).split('.png')[0] + '_1.png'

            f = os.path.join(self.main_dir, filename)
            im = cv2.imread(f, self.imread_mode)
            im = im[:, :, ::-1]

            box = self.find_bbx(m)
            im = im[box[0]:box[2], box[1]:box[3], :]
            m = m[box[0]:box[2], box[1]:box[3]]
            im = cv2.resize(im, (self.maxsize, self.maxsize))
            maskj = cv2.resize(m, (self.maxsize, self.maxsize))
            maskj = np.expand_dims(maskj, axis=-1)
            box = np.array([0, 0, 0, 0])
            # Compute Bbx coordinates

            margin = 3
            bbx = np.zeros_like(maskj)
            xs = np.nonzero(np.sum(maskj, axis=0))[0]
            ys = np.nonzero(np.sum(maskj, axis=1))[0]
            box[1] = xs.min() - margin
            box[3] = xs.max() + margin
            box[0] = ys.min() - margin
            box[2] = ys.max() + margin

            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[3] > self.maxsize: box[3] = self.maxsize-1
            if box[2] > self.maxsize: box[2] = self.maxsize-1

            if box[3] == box[1]:
                box[3] += 1
            if box[0] == box[2]:
                box[2] += 1

            bbx[box[0]:box[2], box[1]:box[3],:] = 1
            # box[2] = box[2] - box[0]
            # box[3] = box[3] - box[1]
            box = box * 1.
            box[0] = box[0]/self.maxsize
            box[2] = box[2] / self.maxsize
            box[1] = box[1] / self.maxsize
            box[3] = box[3] / self.maxsize
            box = np.reshape(box, [1,1,1,4])

            z = np.random.normal(size=[1, 1, 1, self.shapez])
            z2 = np.random.normal(size=[1, 1, 1, self.shapez2])

            yield (im[None, :, :, :], bbx[None, :, :, :], maskj[None, :, :, :], box, z, z2)

    def find_bbx(self, maskj):

        maskj = np.expand_dims(maskj, axis=-1)
        box = np.array([0, 0, 0, 0])

        # Compute Bbx coordinates

        margin = 3
        xs = np.nonzero(np.sum(maskj, axis=0))[0]
        ys = np.nonzero(np.sum(maskj, axis=1))[0]
        box[1] = xs.min() - margin
        box[3] = xs.max() + margin
        box[0] = 0
        box[2] = maskj.shape[0]

        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0

        h = box[2] - box[0]
        w = box[3] - box[1]
        if h < w:
            diff = w - h
            half = int(diff / 2)
            box[0] -= half
            if box[0] < 0:
                box[2] -= box[0]
                box[0] = 0
            else:
                box[2] += diff - half

            if box[2] > maskj.shape[0]:
                box[2] = maskj.shape[0]
        else:
            diff = h - w
            half = int(diff / 2)
            box[1] -= half
            if box[1] < 0:
                box[3] -= box[1]
                box[1] = 0
            else:
                box[3] += diff - half
            if box[3] > maskj.shape[1]:
                box[3] = maskj.shape[1]

        if box[3] == box[1]:
            box[3] += 1
        if box[0] == box[2]:
            box[2] += 1

        return box

class mask_gen_cs_kar():
    """ Produce images read from a list of files. """

    def __init__(self, m_files, main_dir, shape, shapez, shapez2, channel=3):
        """
        Args:
            files (list): list of file paths.
            channel (int): 1 or 3. Will convert grayscale to RGB images if channel==3.
                Will produce (h, w, 1) array if channel==1.
            resize (tuple): int or (h, w) tuple. If given, resize the image.
        """
        assert len(m_files)
        self.m_files = m_files
        self.channel = int(channel)
        assert self.channel in [1, 3], self.channel
        self.imread_mode = cv2.IMREAD_GRAYSCALE if self.channel == 1 else cv2.IMREAD_COLOR
        self.maxsize = shape
        self.shapez = shapez
        self.shapez2 = shapez2
        self.main_dir = main_dir

    def __len__(self):
        return len(self.m_files)

    def __iter__(self):

        for j in self.m_files:
            m = cv2.imread(j, cv2.IMREAD_GRAYSCALE)
            # filename = os.path.basename(j).split('_')[0] + '.jpg'
            # filename = os.path.basename(j).split('.')[0] + '_1.png'
            filename = os.path.basename(j)
            f = os.path.join(self.main_dir, filename)
            im = cv2.imread(f, self.imread_mode)
            im = im[:, :, ::-1]

            box = self.find_bbx(m)
            im = im[box[0]:box[2], box[1]:box[3], :]
            m = m[box[0]:box[2], box[1]:box[3]]
            im = cv2.resize(im, (self.maxsize, self.maxsize))
            maskj = cv2.resize(m, (self.maxsize, self.maxsize))
            maskj = np.expand_dims(maskj, axis=-1)
            box = np.array([0, 0, 0, 0])
            # Compute Bbx coordinates

            margin = 3
            bbx = np.zeros_like(maskj)
            xs = np.nonzero(np.sum(maskj, axis=0))[0]
            ys = np.nonzero(np.sum(maskj, axis=1))[0]
            box[1] = xs.min() - margin
            box[3] = xs.max() + margin
            box[0] = ys.min() - margin
            box[2] = ys.max() + margin

            if box[0] < 0: box[0] = 0
            if box[1] < 0: box[1] = 0
            if box[3] > self.maxsize: box[3] = self.maxsize-1
            if box[2] > self.maxsize: box[2] = self.maxsize-1

            if box[3] == box[1]:
                box[3] += 1
            if box[0] == box[2]:
                box[2] += 1

            bbx[box[0]:box[2], box[1]:box[3],:] = 1
            # box[2] = box[2] - box[0]
            # box[3] = box[3] - box[1]
            box = box * 1.
            box[0] = box[0]/self.maxsize
            box[2] = box[2] / self.maxsize
            box[1] = box[1] / self.maxsize
            box[3] = box[3] / self.maxsize
            box = np.reshape(box, [1,1,1,4])

            z = np.random.normal(size=[1, 1, 1, self.shapez])
            z2 = np.random.normal(size=[1, 1, 1, self.shapez2])

            yield (im[None, :, :, :], bbx[None, :, :, :], maskj[None, :, :, :], box, z, z2)

    def find_bbx(self, maskj):
        resized_width = self.maxsize
        resized_height = self.maxsize
        maskj = np.expand_dims(maskj, axis=-1)

        box = np.array([0, 0, 0, 0])

        # Compute Bbx coordinates

        margin = 3
        xs = np.nonzero(np.sum(maskj, axis=0))[0]
        ys = np.nonzero(np.sum(maskj, axis=1))[0]
        box[1] = xs.min() - margin
        box[3] = xs.max() + margin
        box[0] = ys.min() - margin
        box[2] = ys.max() + margin

        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0

        h = box[2] - box[0]
        w = box[3] - box[1]
        if h < w:
            diff = w - h
            half = int(diff / 2)
            box[0] -= half
            if box[0] < 0:
                box[2] -= box[0]
                box[0] = 0
            else:
                box[2] += diff - half

            if box[2] > maskj.shape[0]:
                box[2] = maskj.shape[0]
        else:
            diff = h - w
            half = int(diff / 2)
            box[1] -= half
            if box[1] < 0:
                box[3] -= box[1]
                box[1] = 0
            else:
                box[3] += diff - half
            if box[3] > maskj.shape[1]:
                box[3] = maskj.shape[1]

        # if box[3] > resized_height: box[3] = resized_height - 1
        # if box[2] > resized_width: box[2] = resized_width - 1

        if box[3] == box[1]:
            box[3] += 1
        if box[0] == box[2]:
            box[2] += 1

        # bbx[box[0]:box[2], box[1]:box[3], :] = 1

        return box

