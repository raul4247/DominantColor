import argparse
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from operator import itemgetter


def histogram(clt):
    # create a histogram based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def sort_color_freq(hist, centroids):
    # hist == freq, and centroids == rgb color
    items = zip(hist, centroids)
    items = sorted(items, key=itemgetter(0), reverse=True)
    return zip(*items)


def print_color(color, percent):
    color = color.astype("uint8").tolist()
    hex = "#{:02x}{:02x}{:02x}".format(color[0], color[1], color[2])

    color = list(map(str, color))
    print('rgb(' + color[0] + ', ' + color[1] + ', ' + color[2] + ')', end='')
    print(' == ', end='')
    print(hex.upper(), end='')
    print(' -> ' + '{0:.2f}'.format(percent*100) + '%')


def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    hist, centroids = sort_color_freq(hist, centroids)

    print('Dominant colors found:')

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color.astype("uint8").tolist(), -1)
        startX = endX
        print_color(color, percent)
        
    return bar


def main():
    # arguments parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Img path")
    ap.add_argument("-c", "--clusters", required=True,
                    type=int, help="Number of clusters")
    args = vars(ap.parse_args())

    # image reading
    print('Parsing image... ', end='')
    src_img = cv2.imread(args["image"])
    # cv2 uses for std bgr on imgs
    print('ok!')

    # reshaping img to list of pixels
    print('Reshaping image... ', end='')
    color_list = src_img.reshape((src_img.shape[0] * src_img.shape[1], 3))
    print('ok!')

    # Applying clustering into colors
    print('Aplying clustering... ', end='')
    clt = KMeans(n_clusters=args["clusters"])
    clt.fit(color_list)
    print('ok!')

    print('Creating histogram and ploting colors... ')
    hist = histogram(clt)
    color_bar = plot_colors(hist, clt.cluster_centers_)

    print('Done!')

    color_bar = cv2.resize(color_bar, (src_img.shape[0], src_img.shape[1]))
    cv2.imshow('Dominant Colors', np.concatenate((src_img, color_bar), axis=1))
    cv2.waitKey()


if __name__ == "__main__":
    main()
