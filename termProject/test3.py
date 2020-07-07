import cv2, numpy as np
import random
## 푸시 다시할라고 추가함

roi_list = list()
model_feature_descriptors = [] # 특징벡터를 저장하는 배열
#model_img = cv2.imread("4.png")
# model_img = cv2.GaussianBlur(model_img, (5, 5), 0)
#model_img = cv2.resize(model_img, dsize=(480, 640), interpolation=cv2.INTER_AREA)
video_path = "test10.mp4"
cap = cv2.VideoCapture(video_path)
_, model_img = cap.read()
MIN_MATCH = 1
dst_pts = []
shape = []
colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (100,0,100), (200,200,200), (0,0,50)]
cap_count = 0
cap_count_list = []

num_features = 2000


# img1 = model_img
#
# orb = cv2.ORB_create(nfeatures=1000)
# keypoints, descriptors = orb.detectAndCompute(img1, None)
#
# img2 = cv2.drawKeypoints(img1, keypoints, None)
#
# cv2.imshow("kp", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def get_model_feature_descriptor():
    try:
        rect = cv2.selectROI('model_img', model_img, fromCenter=False, showCrosshair=False)
        print("rect : ", rect)
        #   rect :  (138, 55, 162, 126)
        tracker = cv2.TrackerMOSSE_create()
        roi = model_img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        print(len(roi))
        return roi
    except:
        return []           # 아무것도 리턴 안돼서 len(roi)가 0이라 roi 지정 종료

def matching(factor) :
    orb = cv2.ORB_create(num_features)
    index_params = dict(algorithm=6,
                        table_number=20,
                        key_size=20,
                        multi_probe_level=2)
    search_params = dict(checks=100)
    global dst_pts
    global cap_count
    global cap_count_list

    while cap.isOpened():
        ret, frame = cap.read()
        res = frame
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        if roi_list[0] is None:  # 등록된 이미지 없음, 카메라 바이패스
            res = frame
        else :
            cap_count = 1
            cnt = 0            # for문을 roi로 돌리길래 인덱스 벡터 cnt
            #model_color = model_color + 1

            for roi in roi_list:
                print(cnt)
                res = frame
                # des1 = model_feature_descriptors[cnt]
                kp1, des1 = orb.detectAndCompute(roi, None)
                kp2, des2 = orb.detectAndCompute(frame, None)

                ratio = 0.75

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                print("!!!!!!!!!!!!!")

                good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * ratio]
                print(good_matches)
                if len(good_matches) > MIN_MATCH * 3:
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])  # 매칭시켜야 하는 물체의 좌표

                mean_point = np.mean(dst_pts, axis=0)
                # mean_y = np.mean(dst_pts, axis=1)

                ## 검출된 특징점 갯수가 일정 숫자 이상이면(지금은 3) 도형정보(shape 배열)랑 매칭해서 도형정보 기
                print("검출된 특징점 개수 : ", len(dst_pts))

                for i in range(len(dst_pts)):  ###### roi와 cap간의 특징벡터 매칭을 통해 나온 cap의 좌표(dst_pts)를 cap영상에 원으로 찍어준다.
                    if len(good_matches) > MIN_MATCH * 5 :
#                        res = cv2.circle(frame, tuple(dst_pts[i]), 3, colors[roi_list.index(roi)], -1)
                        res = cv2.circle(frame, (int(mean_point[0]), int(mean_point[1])), 10, (colors[roi_list.index(roi)]), -1)

                '''
                if matches is not None:
                    for m, n in matches:
                        if m.distance < factor*n.distance:
                            good.append(m)
                '''
                #res = cv2.drawMatches(roi, kp1, frame, kp2, good, res, None, flags=2)
                cnt += 1
            cap_count = 2
        cv2.imshow('Feature Matching', res)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()

def cornerHarris(roi, i) :
    dst = cv2.cornerHarris(roi, 3, 5, 0.07)
    dst = cv2.dilate(dst, None)
    print("코너 갯수 : ", len(dst))     # 코너갯수로 물체 인식할 수도 있지 않을까..

    ###### 검출된 코너 갯수에 따라서 다른 물체로 인식하도록 ####
    # shape 벡터에 순서대로 도형이름 저장
    # roi벡터랑 shape벡터랑 물체가 들어있는 순서가 똑같으니까
    # roi[i]에서 검출된 물체의 모양은 shape[i] 모양임.
    # 일단은 대충만 만든거라 모델영상 바꾸고 코너갯수 다시 찍어본 다음 조건문 수정해야함.
    if len(dst) >= 200 :
        shape.append("menu")
    elif len(dst) > 100 and len(dst) < 200 :
        shape.append("coupon")
    else :
        shape.append(" ")


def main():
    # 특징벡터 :
    cnt = 0
    while True:
        roi = get_model_feature_descriptor()
        if len(roi) != 0 :
            roi_list.append(roi)
            cnt += 1
        else :
            break

    print("number of roi : ", cnt)

    # orb = cv2.ORB_create(num_features)
    # roi 영상에 코너가 몇개 있는지 검출
    for i in range(len(roi_list)):
        cornerHarris(roi_list[i], i)
        # _, des = orb.detectAndCompute(roi_list[i], None)
        # model_feature_descriptors.append(des)

    matching(0.7)

if __name__=="__main__":
    main()



################################################
# 1. 도형정보 검출할 수 있는 꼼수(해리스코너) 추가
# 2. 코너갯수에 따라서 다른 도형(물체)로 인식
# 3. 우측 상단에 갯수만 쓰면 됨
#################################################
