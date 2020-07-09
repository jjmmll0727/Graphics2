import cv2, numpy as np

roi_list = list()
model_feature_descriptors = [] # 특징벡터를 저장하는 배열

video_path = "16.mp4"
cap = cv2.VideoCapture(video_path)
_, model_img = cap.read()
MIN_MATCH = 1
dst_pts = []
shape = []
colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (100,0,100), (200,200,200), (0,0,50)]

cap_count = 0
cap_count_list = []

num_features = 3500

# db에서 값 읽어옴 (좋은 특징점 개수가 각 물체별로 대충 몇개씩인지 적어둠)
def read_data():
    import json
    with open("data2.json", "rt", encoding='UTF-8') as f:
        ret = json.load(f)
        return ret

# roi를 드래그로 지정해서 좌표로 던지는 부분
def get_roi_info():
    try:
        rect = cv2.selectROI('model_img', model_img, fromCenter=False, showCrosshair=False)
        print("rect : ", rect)
        roi = model_img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        print(len(roi))
        return roi
    except:
        return []           # 아무것도 리턴 안돼서 len(roi)가 0이라 roi 지정 종료

# 실제 매칭의 모든 연산을 담당하는 부분
def matching(factor) :
    # <-- db에서 읽어들인 물체별 특징점 갯수의 최대/최소값들을 저장
    data = read_data()
    db_lactofit_min = data['tony']['min']
    db_lactofit_max = data['tony']['max']
    db_book_min = data['coupon']['min']
    db_book_max = data['coupon']['max']
    db_coupon_min = data['charm']['min']
    db_coupon_max = data['charm']['max']
    db_chessboard_min = data['coffee']['min']
    db_chessboard_max = data['coffee']['max']
    # db 읽기 완료 -->

    # orb 알고리즘 사용
    orb = cv2.ORB_create(num_features)
    index_params = dict(algorithm=6,
                        table_number=20,
                        key_size=20,
                        multi_probe_level=2)
    search_params = dict(checks=60)
    global dst_pts
    global cap_count
    global cap_count_list

    while cap.isOpened():
        ret, frame = cap.read()
        res = frame
        if roi_list[0] is None:  # 등록된 이미지 없음, 카메라 바이패스
            res = frame
        else :
            cap_count = 1
            cnt = 0            # for문을 roi로 돌리길래 인덱스 벡터 cnt
            #model_color = model_color + 1

            count_lactofit = 0
            count_book = 0
            count_coupon = 0
            count_chessboard = 0
            print("---------------------------------------")

            for roi in roi_list:
                print(cnt)
                ### <--- roi에서 뽑은 특징점과 영상전체에서 뽑은 특징점들 중 매칭되는것들 추출
                res = frame
                kp1, des1 = orb.detectAndCompute(roi, None)     # roi에서 뽑은 특징점들
                kp2, des2 = orb.detectAndCompute(frame, None)   # 영상전체에서 뽑은 특징점들

                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(des1, des2, k=2)

                good_matches = [m[0] for m in matches \
                                if len(m) == 2 and m[0].distance < m[1].distance * factor]
                dst_pts = []
                if len(good_matches) > MIN_MATCH * 3:
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])  # 매칭시켜야 하는 물체의 좌표

                ### 매칭점 추출 완료 --->

                mean_point = np.mean(dst_pts, axis=0)

                ## 검출된 특징점 갯수가 일정 숫자 이상이면(지금은 3) 도형정보(shape 배열)랑 매칭해서 도형정보 기술
                print("검출된 ", cnt, " 번째 특징점 개수 : ", len(dst_pts))

                if len(good_matches) > MIN_MATCH * 5:
                    if len(dst_pts) >= db_lactofit_min and len(dst_pts) <= db_lactofit_max:
                        count_lactofit = 1
                        print("tony")
                    elif len(dst_pts) >= db_book_min and len(dst_pts) <= db_book_max:
                        count_book = 1
                        print("coupon")
                    elif len(dst_pts) >= db_coupon_min and len(dst_pts) <= db_coupon_max:
                        count_coupon = 1
                        print("charm")
                    elif len(dst_pts) >= db_chessboard_min and len(dst_pts) <= db_chessboard_max:
                        count_chessboard = 1
                        print("coffee")

                ''' 삼각형/사각형/원 (도형)으로 검출내용 표시할때 쓰려고 만든 코드
                    dist_db_circle = abs(len(dst_pts) - db_circle)
                    dist_db_rectangle1 = abs(len(dst_pts) - db_rectangle1)
                    dist_db_rectangle2 = abs(len(dst_pts) - db_rectangle2)
                    dist_db_triangle = abs(len(dst_pts) - db_triangle)
                    which_shape = 0
                    if dist_db_circle < dist_db_rectangle1 and dist_db_circle < dist_db_triangle:
                        count_circles += 1
                        which_shape = 0
                        print("circle")
                    elif dist_db_rectangle1 < dist_db_triangle and dist_db_rectangle1 < dist_db_circle :
                        count_rectangles += 1
                        which_shape = 1
                        print("rectangle")
                    else :
                        count_triangles += 1
                        which_shape = 2
                        print("triangle")
                '''

                for i in range(len(dst_pts)):  # roi와 cap간의 특징벡터 매칭을 통해 나온 cap의 좌표(dst_pts)를 cap영상에 원으로 찍어줌
                    if len(good_matches) > MIN_MATCH * 5 :
                        #res = cv2.circle(frame, (int(mean_point[0]), int(mean_point[1])), 10, (colors[which_shape]), -1)
                        res = cv2.circle(frame, (int(mean_point[0]), int(mean_point[1])), 10, (colors[roi_list.index(roi)]), -1)
                cnt += 1
            cap_count = 2

        # <-- 우측 상단에 검출된 물체 개수 적고 영상 출력
        cv2.putText(res, "tony : " + str(count_lactofit), (450, 30), cv2.FONT_HERSHEY_PLAIN, 1.7, [255, 255, 255], 2)
        cv2.putText(res, "coupon : " + str(count_book), (450, 70), cv2.FONT_HERSHEY_PLAIN, 1.7, [255, 255, 255], 2)
        cv2.putText(res, "charm : " + str(count_coupon), (450, 110), cv2.FONT_HERSHEY_PLAIN, 1.7, [255, 255, 255], 2)
        cv2.putText(res, "coffee : " + str(count_chessboard), (450, 150), cv2.FONT_HERSHEY_PLAIN, 1.7, [255, 255, 255], 2)
        cv2.imshow('Feature Matching', res)
        cv2.waitKey(1)
        # 영상 출력 끝 -->

        #cv2.destroyAllWindows()


def main():
    cnt = 0
    while True:
        roi = get_roi_info()
        if len(roi) != 0 :
            roi_list.append(roi)
            cnt += 1
        else :
            break
    print("number of roi : ", cnt)

    matching(0.75)

if __name__=="__main__":
    main()