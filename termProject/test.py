import cv2
import numpy as np

model_img = cv2.imread("C:\\Users\\jjmml\\Desktop\\models.jpg", cv2.IMREAD_GRAYSCALE)
model_img = cv2.resize(model_img, dsize=(800, 800), interpolation=cv2.INTER_AREA)
video_path = "models_moving.mp4"
cap = cv2.VideoCapture(video_path)
point1 = ()
point2 = ()
drawing = False
model_areas = []       # 모델 도형들의 좌표를 담음
model_feature_descriptors = []


# 스페이스 바 대기
def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)

def mouse_drawing(event, x, y, flags, params) :
    global point1, point2, drawing
    if event == cv2.EVENT_LBUTTONUP :
        if drawing is False :
            drawing = True
            point1 = (x, y)
            point2 = None
        else :
            point2 = (x, y)
            drawing = False
            NW = point1
            SE = point2
            # NE = (point2[0], point1[1])
            # SW = (point1[0], point2[1])
            model_areas.append([NW, SE])


# 모델영상에서 특징점 검출하는 함수
def get_model_feature_descriptor():
    # mouse_handler 함수로 입력받은 영역 내부의 도형에 대해 특징기술자를 추출
    #cv2.namedWindow("MODEL")
    cap = cv2.VideoCapture(video_path)
    while True:
        rect = cv2.selectROI('model_img', model_img, fromCenter=False, showCrosshair=False)
        model_img_tmp = model_img.copy()
        tracker = cv2.TrackerMOSSE_create()

        roi = model_img[rect[1]:rect[1]+rect[2], rect[0]:rect[0]+rect[3]]
        cv2.imshow("roi", roi)
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(roi, None)
        model_feature_descriptors.append(des)
        print(model_feature_descriptors)
        ORB(model_feature_descriptors,cap)


def ORB(model_img, cap):
    orb = cv2.ORB_create()
    while True:
        cap = cv2.VideoCapture(video_path) # cap은 video의 매순간 순간을 캡쳐
        _, cap = cap.read()

        #orb.setHessianThreshold(400)
        # find the keypoints and descriptors with SIFT
        #kp1, des1 = orb.detectAndCompute(model_img, None)
        kp1, des1 = orb.detectAndCompute(model_img, None)
        kp2, des2 = orb.detectAndCompute(cap, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawKeypoints(cap, kp2, None, (255,0,0), flags=0)
        cv2.imshow("img3", img3)

        '''
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
        #img3 = cv2.drawMatchesKnn(model_img, kp1, cap, kp2, matches, None, **draw_params)
        #img3 = cv2.drawKeypoints(cap, kp2, img3)
        print("here")
        cv2.imshow("img3", img3)
        return img3
        '''

'''
def matching():
    while True:
        ret, img = cap.read()
        success, box = tracker.update(img)

        # if success:
        left, top, w, h = [int(v) for v in box]
        right = left + w
        bottom = top + h
        center = (int(left + w / 2), int(top + h / 2))
        radian = (int(w / 2), int(h / 2))

        cv2.ellipse(img, center, radian, 0, 0, 360, (0, 255, 255), 3)
        cv2.imshow('playing', img)
    # 두 특징기술자(벡터들)을 비교하여 매칭여부 판단
    # 매칭된 도형은 마커로 마킹, 좌상단에 검출된 도형의 갯수를 입력

'''
def main():
    ############################################################
    # 입력 영상 최적화 할거면 여기서
    ############################################################

    #get_model_feature_descriptor()
    ORB(model_img, cap)
    #matching()
    #cv2.imshow("www", ORB(cap, model_img))# 1. 입력모델(mode_shapes.jpg, 'l'오타남)에서 특징점 검출 -> model_feature_descriptors 배열에 저장
    # get_target_feature_descriptor()    # 2. 캠영상 입력받아서 실시간으로 특징점 검출

main()


# 한거
# 1. 모델영상에서 특징점 검출 알고리즘 돌릴 roi 분리 (마우스로 좌상단 / 우하단 클릭하면 직사각형 생김)
# 2.


# 해야할 작업
# 1. 모델영상에서 특징기술자 검출해서 각 특징기술자가 어떤 도형에 해당되는지 미리 매핑(db화)
# 2. 빈 코드 채우기