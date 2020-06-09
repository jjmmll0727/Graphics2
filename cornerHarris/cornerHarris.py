import cv2
import numpy as np

filename = 'chess2.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,7,11,0.04)
# ksize 가 클수록 특징점으로 추출되는 값이 많아져
# k가 작을수록 더 많이 추출된다
# block size가 클수록 더 많이 추출된다
# --> 더 많이 추출된다는 의미는 정해진 임계값보다 큰 특징 가능성 값이 많다는 뜻 + 이웃화소 보다 큰 값이 많다는 뜻

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
cv2.waitKey()
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()