import cv2
import numpy as np
def feature():
    filename = 'butterfly.jpg'
    img = cv2.imread(filename)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main():
    feature()

if __name__=="__main__":
    main()