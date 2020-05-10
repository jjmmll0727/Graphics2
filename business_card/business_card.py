import cv2
import numpy as np
from matplotlib import pyplot as plt
clicked = []

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        clicked.append([x, y])
        print(clicked)

    img = cv2.imread('car.jpg')
    cv2.imshow('img', img)

def nothing(x):
    pass

def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)

def grabcut(resized):
    mask = np.zeros(resized.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, 560, 460)
    cv2.grabCut(resized, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    grabcutted = resized * mask2[:, :, np.newaxis]

    cv2.imshow('grabcut', grabcutted)
    return grabcutted

def canny(grabcutted):
    gray = cv2.cvtColor(grabcutted, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)
    cv2.imshow('canny', img_canny)
    return img_canny

def findingLine(img_canny):
    src = cv2.imread('card2.jpg')
    lines = cv2.HoughLines(img_canny, 0.8, np.pi / 180, 150, srn = 100, stn = 200, min_theta = 0, max_theta = np.pi)
    for i in lines:
        rho, theta = i[0][0], i[0][1]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho

        scale = src.shape[0] + src.shape[1]

        x1 = int(x0 + scale * -b)
        y1 = int(y0 + scale * a)
        x2 = int(x0 - scale * -b)
        y2 = int(y0 - scale * a)

        cv2.line(img_canny, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(img_canny, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

    cv2.imshow('findline', img_canny)

def main():
    ori_img = cv2.imread('card2.jpg')
    resized = cv2.resize(ori_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)

    cv2.imshow('resize', resized)
    wait()
    grabcut(resized)
    wait()
    canny(grabcut(resized))
    wait()
    findingLine(canny(grabcut(resized)))
    wait()


if __name__ == "__main__":
    main()

