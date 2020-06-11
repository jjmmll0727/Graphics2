import cv2, numpy as np


model_img = cv2.imread("test.png")
#model_img = cv2.resize(model_img, dsize=(700, 700), interpolation=cv2.INTER_AREA)
img1 = model_img
win_name = 'Camera Matching'
MIN_MATCH = 1 # 최소한의 매칭 기준
# ORB 검출기 생성  ---①
detector = cv2.ORB_create(1000)
# Flann 추출기 생성 ---②
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6,
                   key_size = 12,
                   multi_probe_level = 1)
search_params=dict(checks=60) # 계산 반복 횟수 횟수가 증가할수록 정확도는 높아지고 속도는 떨어져
matcher = cv2.FlannBasedMatcher(index_params, search_params)
# 카메라 캡쳐 연결 및 프레임 크기 축소 ---③
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
model_feature_descriptors = [] # 특징벡터를 저장하는 배열
roi_list = [] # roi들을 담을 배열 --> roi에서 keypoint와 특징벡터를 뽑아내기 위해
dst_pts = []
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 모델영상에서 특징점 검출하는 함수 여기서는 grayscale아니어도 상관없어
def get_model_feature_descriptor():
    # mouse_handler 함수로 입력받은 영역 내부의 도형에 대해 특징기술자를 추출
    #cv2.namedWindow("MODEL")
    global model_feature_descriptors
    cap = cv2.VideoCapture(video_path)
    for i in range(4):
        rect = cv2.selectROI('model_img', model_img, fromCenter=False, showCrosshair=False)
        #model_img_tmp = cv2.cvtColor(model_img, cv2.COLOR_BGR2GRAY)
        tracker = cv2.TrackerMOSSE_create()

        roi = model_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        cv2.imshow("roi", roi)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_list.append(roi) # roi를 저장할때 grayscale인 상태이다,
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(roi, None)

        model_feature_descriptors.append(des)
        print(model_feature_descriptors)
        print(roi_list)
        key = cv2.waitKey(1)

# 특징벡터 배열과 cap이미지를 비교매칭해야 한다. 그래서 매칭을 하면 매칭점만을 cap화면에 보여주게 하자 일단
def matching():
    global roi_list
    global dst_pts
    dst_pts_x = 0
    dst_pts_y = 0
    dst_list = []
    res = None
    while cap.isOpened():
        ret, frame = cap.read()
        img1 = model_img # 1. 원본이미지 --> 특징벡터가 담긴 배열의 성분으로 대체 해야 돼
        if roi_list[0] is None:  # 등록된 이미지 없음, 카메라 바이패스
            res = frame
        else:             # 등록된 이미지 있는 경우, 매칭 시작
            img2 = frame  # 2. cap 이미지
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            # 키포인트와 디스크립터 추출
            kp1, desc1 = detector.detectAndCompute(roi_list[0], None)
            kp2, desc2 = detector.detectAndCompute(gray2, None)


            # k=2로 knnMatch
            matches = matcher.knnMatch(desc1, desc2, 2)

            # 이웃 거리의 75%로 좋은 매칭점 추출---②
            ratio = 0.50
            good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * ratio]
            print('good matches:%d/%d' %(len(good_matches),len(matches)))
            # 모든 매칭점 그리지 못하게 마스크를 0으로 채움
            matchesMask = np.zeros(len(good_matches)).tolist()
            # 좋은 매칭점 최소 갯수 이상 인 경우
            if len(good_matches) > MIN_MATCH*1:
                # 좋은 매칭점으로 원본과 대상 영상의 좌표 구하기 ---③
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ])
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]) # 매칭시켜야 하는 물체의 좌표
                #print("-----------------")
                #print(dst_pts)
                # 원근 변환 행렬 구하기 ---⑤
                mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                accuracy=float(mask.sum()) / mask.size
                print("accuracy: %d/%d(%.2f%%)"% (mask.sum(), mask.size, accuracy*100))

                if mask.sum() > MIN_MATCH:  # 정상치 매칭점 최소 갯수 이상 인 경우
                    # 이상점 매칭점만 그리게 마스크 설정
                    matchesMask = mask.ravel().tolist()
                    # 원본 영상 좌표로 원근 변환 후 영역 표시  ---⑦
                    h,w, = roi_list[0].shape[:2]
                    pts = np.float32([ [[0,0]],[[0,h-1]],[[w-1,h-1]],[[w-1,0]] ])
                    dst = cv2.perspectiveTransform(pts,mtrx)
                    #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
            # 마스크로 매칭점 그리기 ---⑨

            #res = cv2.drawMatches(roi_list[0], kp1, img2, kp2, good_matches, None, \
            #                    matchesMask=matchesMask,
            #                   flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

            for i in range(len(dst_pts)): # roi와 cap간의 특징벡터 매칭을 통해 나온 cap의 좌표(dst_pts)를 cap영상에 원으로 찍어준다.
                res = cv2.circle(img2, tuple(dst_pts[i]), 3, (0,255,0), -1)
                dst_pts_x = dst_pts_x + dst_pts[i][0]
                dst_pts_y = dst_pts_y + dst_pts[i][1]

            dst_pts_x = dst_pts_x / len(dst_pts)
            dst_pts_y = dst_pts_y / len(dst_pts)
            dst_list.append(dst_pts_x)
            dst_list.append(dst_pts_y)
        # 결과 출력
        cv2.imshow(win_name, res)

        key = cv2.waitKey(1)
        if key == 32:    # Esc, 종료
                break
        elif key == ord(' '): # 스페이스 바로 ROI 설정해서 img1 설정
            x,y,w,h = cv2.selectROI(win_name, frame, False)
            if w and h:
                roi_list[0] = frame[y:y+h, x:x+w]
    else:
        print("can't open camera.")
    cap.release()
    cv2.destroyAllWindows()

def main():
    get_model_feature_descriptor()
    matching()


if __name__=="__main__":
    main()
