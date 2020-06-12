import cv2, numpy as np

roi_list = list()
model_feature_descriptors = [] # 특징벡터를 저장하는 배열
model_img = cv2.imread("test.png")
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
MIN_MATCH = 1
dst_pts = []
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
        return roi
    except:
        return []           # 아무것도 리턴 안돼서 len(roi)가 0이라 roi 지정 종료

def matching(factor) :
    orb = cv2.ORB_create(nfeatures=1000)
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=40)
    global dst_pts

    while cap.isOpened():
        ret, frame = cap.read()
        if roi_list[0] is None:  # 등록된 이미지 없음, 카메라 바이패스
            res = frame
        else :
            for roi in roi_list:
                res = None
                kp1, des1 = orb.detectAndCompute(roi, None)
                kp2, des2 = orb.detectAndCompute(frame, None)
                FLANN_INDEX_KDTREE = 0

                good = []
                ratio = 0.50

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)
                good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * ratio]
                if len(good_matches) > MIN_MATCH * 1:
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])  # 매칭시켜야 하는 물체의 좌표

                for i in range(len(dst_pts)):  ###### roi와 cap간의 특징벡터 매칭을 통해 나온 cap의 좌표(dst_pts)를 cap영상에 원으로 찍어준다.
                    res = cv2.circle(frame, tuple(dst_pts[i]), 3, (0, 255, 0), -1)
                '''
                if matches is not None:
                    for m, n in matches:
                        if m.distance < factor*n.distance:
                            good.append(m)
                '''
                #res = cv2.drawMatches(roi, kp1, frame, kp2, good, res, None, flags=2)
        cv2.imshow('Feature Matching', res)
        cv2.waitKey(1)
        #cv2.destroyAllWindows()


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

    matching(0.7)

    # roi = get_model_feature_descriptor()
    # orb = cv2.ORB_create(nfeatures=300)
    # keypoints, descriptors = orb.detectAndCompute(roi, None)
    # result_image = cv2.drawKeypoints(roi, keypoints, None)

    # cv2.imshow("kp", result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # matching()


if __name__=="__main__":
    main()