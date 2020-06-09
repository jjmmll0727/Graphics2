import cv2
import numpy as np

model_img = cv2.imread('mode_shapes.jpg', cv2.IMREAD_GRAYSCALE)
point1 = ()
point2 = ()
drawing = False
model_areas = []       # 모델 도형들의 좌표를 담음
model_feature_descriptors = []

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
    cv2.namedWindow("MODEL")
    cv2.setMouseCallback("MODEL", mouse_drawing)
    while True:
        model_img_tmp = model_img
        if point1 and point2:
            cv2.rectangle(model_img_tmp, point1, point2, (0, 255, 0))
        cv2.imshow("MODEL", model_img_tmp)
        if cv2.waitKey(1) == 27:
            print(model_areas)
            break

    for i in range(len(model_areas)):
        roi = model_img[model_areas[i][0][1]+3:model_areas[i][1][1], model_areas[i][0][0]+3:model_areas[i][1][0]]
        # cv2.imshow("roi", roi)      # 모델 영상에서 각 도형 부분만 하나씩 떼서 나옴.
        #                             # 3씩 더한 이유는 안더했더니 cv2.rectangle로 그렸던 선까지 같이 나와서
        # cv2.waitKey()
        ############################### 여기서부터 특징점 검출 알고리즘 ##################################
        # 1. for문을 도는 roi가 모델 떼어넨것. sift / orb / surf 등등 써서 특징점 검출해야함
        # 2. model_feature_descriptior 배열에 특징벡터 담아주세요

        ##########################################################################################


# 캠화면(동영상)에서 특징점 검출하는 함수
def get_target_feature_descriptor():
    # 캠 영상을 입력받아 영상의 특징기술자를 추출
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        ################################# 여기서부터 특징점 검출 알고리즘 #################################
        # 1. 특징점 검출
            # 2. 매칭 => 함수 matching()
        # 3. 좌측 상단에 매칭된 도형 개수 출력
        ###########################################################################################

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def matching():
    # 두 특징기술자(벡터들)을 비교하여 매칭여부 판단
    # 매칭된 도형은 마커로 마킹, 좌상단에 검출된 도형의 갯수를 입력
    pass

def main():
    ############################################################
    # 입력 영상 최적화 할거면 여기서
    ############################################################
    get_model_feature_descriptor()     # 1. 입력모델(mode_shapes.jpg, 'l'오타남)에서 특징점 검출 -> model_feature_descriptors 배열에 저장
    # get_target_feature_descriptor()    # 2. 캠영상 입력받아서 실시간으로 특징점 검출

main()


# 한거
# 1. 모델영상에서 특징점 검출 알고리즘 돌릴 roi 분리 (마우스로 좌상단 / 우하단 클릭하면 직사각형 생김)
# 2.


# 해야할 작업
# 1. 모델영상에서 특징기술자 검출해서 각 특징기술자가 어떤 도형에 해당되는지 미리 매핑(db화)
# 2. 빈 코드 채우기  