import argparse
import cv2
import matplotlib.pyplot as plt

def show_img(img):
    plt.figure()
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def main():
    # arguments parsing
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Img path")
    args = vars(ap.parse_args())

    # image reading
    print('Parsing image... ', end='')
    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2 uses for std bgr on imgs
    print('ok!')

    show_img(image)


if __name__ == "__main__":
    main()