import numpy as np
import cv2

# 사용환경 : opencv-contrib-python -3.4.2.17 version
# 사용법 : 객체를 드레그한 뒤 스페이스바를 누른다.
#        동영상 재생될때 q누르면 종료 s누르면 다른 객체 식별.

errorcode = 0
cnt = 0
model_img = cv2.imread("C:/Users/jjmml/Desktop/models.jpg", cv2.IMREAD_GRAYSCALE)
model_img = cv2.resize(model_img, dsize=(800, 800), interpolation=cv2.INTER_AREA)

def select(cap):
    global errorcode
    global cnt
    tracker = cv2.TrackerMOSSE_create()
    ret, img = cap.read()
    cv2.namedWindow('Select Window')

    cv2.putText(img, 'Usage: Drag Mouse and Press a Spacebar', (10, 30), cv2.FONT_ITALIC, 1, (35, 30, 100), 2,
                cv2.LINE_AA)
    cv2.putText(img, 'Esc: Exit', (10, 70), cv2.FONT_ITALIC, 1, (35, 30, 100), 2, cv2.LINE_AA)

    cv2.imshow('Select Window', img)

    # select ROI #### rect = (왼쪽 좌표(x좌표), 위쪽 좌표(y좌표), 너비, 높이)로 구성
    rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=False)
    # rect부분 이미지 출력
    dst = img.copy()
    dst = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]  # 이미지 자르기
    # cv2.imshow('blue',dst)

    cv2.destroyWindow('Select Window')

    # initialize tracker
    try:
        tracker.init(img, rect)
        print(tracker)
    except:
        cv2.destroyAllWindows()
        cap.release()
    return tracker, dst, rect[2], rect[3]


def ORB(model_img, cap):
    global errorcode
    global cnt
    while True:
        img1 = model_img  # queryImage
        _, img2 = cap.read()  # trainImage

        # Initiate SURF detector
        #surf = cv2.xfeatures2d.SURF_create()
        orb = cv2.ORB_create()
        #orb.setHessianThreshold(400)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        try:
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            # ratio test as per Loew's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = dict(matchColor=(0, 0, 255),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=0)
            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
            print(matchesMask)
            return img3
        except:
            errorcode = 1
            cnt = 1
            # cv2.putText(img,'nothing',(10,30),cv2.FONT_ITALIC,1,(0,0,255),2,cv2.LINE_AA)
            # cap.release()
            return 1


# open video file
video_path = "models_moving.mp4"
cap = cv2.VideoCapture(video_path)


# global variables

count = 0

tracker, dst, num1, num2 = select(cap)
output_size = (num1 + 100, num2 + 100)  # result_img 의 가로, 세로 크기 조정


while True:
    count += 1
    # read frame from video
    ret, img = cap.read()
    ORB(model_img, cap)

    if not ret:
        break

        # update tracker and get position from new frame
    success, box = tracker.update(img)

    # if success:
    left, top, w, h = [int(v) for v in box]
    right = left + w
    bottom = top + h
    center = (int(left + w / 2), int(top + h / 2))
    radian = (int(w / 2), int(h / 2))

    cv2.ellipse(img, center, radian, 0, 0, 360, (0, 255, 255), 3)


    cv2.imshow('playing', img)
    #cv2.imshow('result', result_img)
    #img3 = ORB(dst, result_img)

    try:
        if errorcode == 0 and cnt == 0:
            # cv2.imshow('img3', img3)
            print("gggggggggggggg")
        else:
            cv2.destroyWindow('matching')
    except:
        break

    key = cv2.waitKey(1)

# release everything
cv2.destroyAllWindows()
cap.release()




