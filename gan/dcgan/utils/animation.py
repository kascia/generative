import pylab
import imageio
from glob import glob
import re
import numpy as np
from scipy.misc import imresize

class Animation(object):

    def __init__(self, shuffle=True, height=32, width=32):
        imageio.plugins.ffmpeg.download()
        self.dataset_dir = '/data/leewk92/generative/animation/sg02.mp4'
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.offset = 0
        self.dataset = self.load_dataset()


    def load_dataset(self):
        vid = imageio.get_reader(self.dataset_dir,  'ffmpeg')
        #self.total_num = vid.get_length()
        self.total_num = 5000
        images = []
        for i in range(self.total_num):
            if i % 1000 == 0:
                print(i)
            image = vid.get_data(i)
            image = self.normalize_image(image)
            images.append(image)

        images = np.array(images)
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

