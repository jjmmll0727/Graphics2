import cv2
import numpy as np
from matplotlib import pyplot as plt
clicked = [] #마우스 포인터 위치 저장 --> grabcut 할때 씌여 (일반화를 위해)

fileName = "unnamed2.jpg"

# 스페이스 바 대기
def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        clicked.append([x, y])
        print(clicked)
def mousePointer(resized):
    cv2.namedWindow("mousePointer")
    cv2.setMouseCallback("mousePointer", mouse_handler, param = resized)

    while True:
        cv2.imshow("mousePointer", resized)
        k = cv2.waitKey(1) & 0xFF

        if k == 32:
            break
    #cv2.destroyAllWindows()

# 명함 바깥부분에 직선형태의 shape이 있어서 다 제거해주기 위해 grabcut을 함
def grabcut(resized):
    mask = np.zeros(resized.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    height, width, channel = resized.shape
    rect = (clicked[0][0], clicked[0][1], clicked[1][0], clicked[1][1]) #명함을 다 포함할수 있는 수평 수직의 직사각형의 좌상 우하 꼭짓점
    cv2.grabCut(resized, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    grabcutted = resized * mask2[:, :, np.newaxis]

    #cv2.imshow('grabcut', grabcutted)
    return grabcutted

## 모폴로지 연산을 위한 커널
# 모폴로지 연산 이유: 배경과 전경의 밝기 차이가 적어 grabcut 함수를 사용하고도 굴곡을 지울수 없었다.
# 최대한 굴곡을 지우기 위해 모폴로지 연산을 수행함.
## 모폴로지 연산 closing 연산횟수: 10 그 이상 그 이하로 하면 형태가 뭉개져

def canny(grabcutted):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = cv2.cvtColor(grabcutted, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0) # 한번더 굴곡을 줄이기 위해서 블러 처리함

    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=10)
    img_canny = cv2.Canny(closing, 5000, 1500, apertureSize = 5, L2gradient = True)
    return img_canny
    # canny 연산을 할때 이진 모폴로지와 가우시안 스무딩 까지 해서 명함안의 잡음 제거

def order_points(pts):
    rect = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmin(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmin(diff)]

    return rect


# 네변을 그려주고 꼭짓점을 찍은후, 투영변환까지 해주는 함수
def drawlingCorrectly(img_canny):

    src = cv2.imread(fileName)

    resized = cv2.resize(src, dsize=(640,480), interpolation=cv2.INTER_AREA)
    dst = resized.copy()

    # 2. 선분 검출을 적용하여 4변을 추출
    ( cnts, _) = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    cv2.drawContours(img_canny, [cnt], 0, (255, 255, 0), 1)
    epsilon = 0.03*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True) # 들어가는 순서가 자기마음대로

    size = len(approx) # 4변의 사각형임을 알수 있어

    # cv2.line(dst, tuple(approx[0][0]), tuple(approx[size-1][0]), (0,255,0), 3)

    cv2.line(dst, tuple(approx[0][0]), tuple(approx[1][0]), (0, 255, 0), 3) # topleft부터 대입해서 approx[0][0] --> topleft
    cv2.line(dst, tuple(approx[1][0]), tuple(approx[2][0]), (0, 0, 255), 3)
    cv2.line(dst, tuple(approx[2][0]), tuple(approx[3][0]), (255, 0, 0), 3)
    cv2.line(dst, tuple(approx[3][0]), tuple(approx[0][0]), (255, 255, 0), 3)

    approx = approx.reshape(4,2)
    (topLeft, topRight, bottomRight, bottomLeft) = approx

    '''
    w1 = abs(bottomRight[0] - topRight[0])
    w2 = abs(topLeft[0] - bottomLeft[0])
    h1 = abs(topLeft[1] - topRight[1])
    h2 = abs(bottomLeft[1] - bottomRight[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    '''

    #   3.전단계에서 추출한 네 선분을 이용하여 명함영역의 네 꼭지점을 계산한다.
    #   4좌표를 sorting 하고 4좌표끼리의 대소비교를 하여 src에 대입한다. jpg 파일마다 순서가 바뀌므로 sorting해서 일반화를 해야해

    # 투영변횐
    src = []

    src.append([float(bottomRight[0]), float(bottomRight[1])]) # 좌상 부터 시계방향으로 대입
    src.append([float(topRight[0]), float(topRight[1])])
    src.append([float(topLeft[0]), float(topLeft[1])])
    src.append([float(bottomLeft[0]), float(bottomLeft[1])])

    src.sort(key = lambda x:x[0]) #이차원배열을 x성분을 기준으로 sort
    src_tmp = [] # 좌상, 좌하, 우상, 우하 순서대로 sorting 해서 저장 .. 이게 real src

    # 가장 작은 두 x값에 해당되는 좌표의 y값들을 비교
    if src[0][1] < src[1][1]: #sort의 두번째 원소의 y값이 더 큰 경우
        src_tmp.append(src[0])
        src_tmp.append(src[1])
    else:
        src_tmp.append(src[1])
        src_tmp.append(src[0])

    # 가장 큰 두 x값에 해당되는 좌표의 y값들을 비교
    if src[2][1] < src[3][1]:
        src_tmp.append(src[2])
        src_tmp.append(src[3])
    else:
        src_tmp.append(src[3])
        src_tmp.append(src[2])

    # 4좌표에 점찍기 (좌상 좌하 우상 우하)
    cv2.circle(dst, (int(src_tmp[0][0]), int(src_tmp[0][1])), 4, (0, 0, 0), -1)
    cv2.circle(dst, (int(src_tmp[1][0]), int(src_tmp[1][1])), 4, (255, 0, 255), -1)
    cv2.circle(dst, (int(src_tmp[2][0]), int(src_tmp[2][1])), 4, (0, 255, 255), -1)
    cv2.circle(dst, (int(src_tmp[3][0]), int(src_tmp[3][1])), 4, (100, 100, 0), -1)

    # 추출된 네변(선분), (즉, 좌, 우, 상, 하단 )의 기울기, y 절편, 양끝점의 좌표을 각각 출력할 것.
    pos1 = [0, 3]
    pos2 = [1, 2]
    for first in pos1:
        for second in pos2:
            a = (src_tmp[first][1]-src_tmp[second][1])/(src_tmp[first][0]-src_tmp[second][0])   # 기울기
            b = (-a)*src_tmp[first][0] + src_tmp[first][1]  # 절편
            print("# 기울기 : ", a, "   절편 : ", b, "    양 끝점 : ", src_tmp[first], src_tmp[second])


    # 4좌표값 출력
    print("")
    print("좌상단 꼭지점 : ", src_tmp[0])
    print("좌하단 꼭지점 : ", src_tmp[1])
    print("우상단 꼭지점 : ", src_tmp[2])
    print("우하단 꼭지점 : ", src_tmp[3])

    w1 = abs(src_tmp[0][0] - src_tmp[2][0])
    w2 = abs(src_tmp[1][0] - src_tmp[3][0])
    h1 = abs(src_tmp[0][1] - src_tmp[1][1])
    h2 = abs(src_tmp[2][1] - src_tmp[3][1])
    maxWidth = int(max([w1, w2]))
    maxHeight = int(max([h1, h2]))

    src_np = np.array(src_tmp, dtype=np.float32)

    # 4. 네 꼭지점을 이용하여 명함영역이 직사각형이 되도록 기하변환 한다.
        # 좌상 좌하 우상 우하 순서대로 mapping
    dst_np = np.array([
        [0, 0],
        [0, maxHeight],
        [maxWidth, 0],
        [maxWidth, maxHeight]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src=src_np, dst=dst_np)
    perspective = cv2.warpPerspective(resized, M=M, dsize=(maxWidth, maxHeight))

    cv2.imshow('drawlingline', dst)
    wait()

    cv2.imshow('perspective', perspective)

def transition(img, positions) :
    dst = np.array([[0,0], [0, 300], [400, 0], [400, 300]], dtype=np.float32)
    kernel = cv2.getPerspectiveTransform(src=positions, dst=dst)
    print("변환행렬 : ", kernel)
    result = cv2.warpPerspective(img, M=kernel, dsize=(400, 300))
    cv2.imshow('result', result)
    return result

def main():
    ori_img = cv2.imread(fileName)
    resized = cv2.resize(ori_img, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    # 영상의 사이즈가 너무 커서 조정함

    # 0. 원본 이미지
    #cv2.imshow('resize', resized)
    mousePointer(resized)


    # 1. 원본 + 에지 출력
    img_canny = canny(grabcut(resized))
    img_canny_output = cv2.merge((img_canny, img_canny, img_canny))
    original_image_with_edge_image = cv2.hconcat([resized, img_canny_output])
    cv2.imshow('1. original_image, edge_image', original_image_with_edge_image)
    wait()

    # 2 ~ 4
    drawlingCorrectly(img_canny)
    wait()


if __name__ == "__main__":
    main()

