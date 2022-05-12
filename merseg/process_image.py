import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage.draw as skimg
import skimage.io as skio
import os


def draw_ellipse_overlay_synthetic(objs, image, name="overlay.TIFF"):
    for o in objs:
        row, col = skimg.ellipse_perimeter(int(o.centre.x), int(o.centre.y),
                                           int(o.shape.minor), int(o.shape.major), (o.shape.rotation * -1),
                                           shape=image.shape)
        image[row, col] = [0, 125, 0]

    skio.imsave(os.path.join(os.getcwd(), 'output', name), image, check_contrast = False)




class ProcessImage(object):
    def __init__(self):
        self.X = 0
        self.Y = 0
        self.threshold_val = 0.0
        self.img_greyscale = None
        self.image_RGB = None
        self.fn = ""

    def compute_probs(self, fn):
        pxls = self._read_in_img(fn)
        front_bins, back_bins = self._k_means_bins(pxls)
        X = self.X
        Y = self.Y

        self.fn = fn

        fore_probs = np.zeros((X * Y))
        back_probs = np.zeros((X * Y))
        pxls = pxls.flatten()

        for i in range(len(pxls)):
            if pxls[i] > self.threshold_val:
                fore_probs[i] = np.log(front_bins[int(pxls[i])])
                back_probs[i] = 0
            else:
                back_probs[i] = np.log(back_bins[int(pxls[i])])
                fore_probs[i] = 0

        fore_probs.resize((X, Y))
        back_probs.resize((X, Y))
        return fore_probs, back_probs

    def produce_binary_img(self, fn):
        if self.fn != fn:
            pxls = self._read_in_img(fn)
            self._k_means_bins(pxls)
            self.fn = fn

        val, binary = cv2.threshold(self.img_greyscale, self.threshold_val, 1, cv2.THRESH_BINARY)
        return binary

    def draw_ellipse_overlay(self, objs, name="overlay.png", image=None):
        if not image:
            image = self.image_RGB
        for o in objs:
            row, col = skimg.ellipse_perimeter(int(o.centre.x), int(o.centre.y),
                                               int(o.shape.minor), int(o.shape.major), (o.shape.rotation * -1),
                                               shape=image.shape)
            image[row, col] = [0, 125, 0]

        plt.imshow(image)
        plt.imsave(name, image)

    def _read_in_img(self, fn):
        image = cv2.imread(fn)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pxls = img.reshape((-1, 1))
        pxls = np.float32(pxls)
        self.Y, self.X = img.shape[::-1]
        self.img_greyscale = img
        self.image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return pxls

    def _k_means_bins(self, pxls):
        criteria = (cv2.TERM_CRITERIA_EPS, 500, 0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(pxls, 2, None, criteria, 10, flags)
        back = pxls[labels == 0]
        fore = pxls[labels == 1]
        back_bins = plt.hist(back, 256, [0, 256], density=True)
        front_bins = plt.hist(fore, 256, [0, 256], density=True)
        back_bins = back_bins[0]
        front_bins = front_bins[0]
        self.threshold_val = np.amax(back)
        plt.clf()
        return front_bins, back_bins