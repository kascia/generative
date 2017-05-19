from glob import glob
import re
import numpy as np
from matplotlib import pyplot
from scipy.misc import imresize


class YaleFace(object):

    def __init__(self, shuffle=True, height=32, width=32):
        self.dataset_dir = '/data/leewk92/generative/yaleface/'
        self.pgm_list = glob(self.dataset_dir+'*/*.pgm')
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.offset = 0
        self.dataset = self.load_dataset()

    def load_dataset(self):
        images = []

        for pgm_filename in self.pgm_list:
            try:
                image = self.read_pgm(pgm_filename)
                image = self.normalize_image(image)
                images.append(image)
            except:
                print('file reading is failed', pgm_filename)

        images = np.array(images)
        self.total_num = len(images)
        if self.shuffle:
            perm = np.arange(self.total_num)
            np.random.shuffle(perm)
            images = images[perm]
        return images


    def normalize_image(self, image):
        image = imresize(image, (self.height, self.width))
        image = np.array(image, dtype=np.float32)
        image /=127.5
        image -= 1.0
        return image


    def read_pgm(self, filename, byteorder='>'):
        """Return image data from a raw PGM file as numpy array.

        Format specification: http://netpbm.sourceforge.net/doc/pgm.html

        """
        with open(filename, 'rb') as f:
            buffer = f.read()
        try:
            header, width, height, maxval = re.search(
                b"(^P5\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n])*"
                b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)
        return np.frombuffer(buffer,
                                dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                                count=int(width)*int(height),
                                offset=len(header)
                                ).reshape((int(height), int(width)))


    def next_batch(self, batch_size):
        total_num = self.total_num
        if batch_size > total_num:
            print('error : batch_size > total_num')
            return
        offset = self.offset
        until = offset + batch_size
        if until >= total_num:
            until -= total_num
            batch = np.concatenate((self.dataset[offset:], self.dataset[:until]))
        else:
            batch = self.dataset[offset:until]
        self.offset = until
        return batch