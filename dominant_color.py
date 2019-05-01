import argparse
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from operator import itemgetter
from Color import Color
from Util import verbose_print


def histogram(km):
    numLabels = np.arange(0, len(np.unique(km.labels_)) + 1)
    (hist, _) = np.histogram(km.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def sort_color_freq(freq, colors):
    items = zip(freq, colors)
    items = sorted(items, key=itemgetter(0), reverse=True)
    return zip(*items)


def get_colors(hist, centroids):
    hist, centroids = sort_color_freq(hist, centroids)
    colors = []

    for (percent, color) in zip(hist, centroids):
        colors.append(Color(color.astype("uint8").tolist(), percent))

    return colors


def plot_colors(colors):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for color in colors:
        endX = startX + (color.freq * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.rgb, -1)
        startX = endX
        
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Img path")
    ap.add_argument("-c", "--clusters", required=True, type=int, help="Number of clusters")
    ap.add_argument("-v", "--verbose", required=False, help="Verbose Mode")
    args = vars(ap.parse_args())

    verbose_print('Verbose mode on...', args["verbose"])

    verbose_print('Parsing image... ', args["verbose"])
    src_img = cv2.imread(args["image"])

    verbose_print('Reshaping img to list of pixels... ', args["verbose"])
    color_list = src_img.reshape((src_img.shape[0] * src_img.shape[1], 3))

    verbose_print('Applying clustering to colors... ', args["verbose"])
    km = KMeans(n_clusters=args["clusters"])
    km.fit(color_list)

    verbose_print('Creating histogram...', args["verbose"])
    hist = histogram(km)
    colors = get_colors(hist, km.cluster_centers_)
    for c in colors:
        print(c)

    verbose_print('Ploting colors...', args["verbose"])
    color_bar = plot_colors(colors)
    color_bar = cv2.resize(color_bar, (src_img.shape[0], src_img.shape[1]))
    cv2.imshow('Dominant Colors', np.concatenate((src_img, color_bar), axis=1))
    cv2.waitKey()


if __name__ == "__main__":
    main()