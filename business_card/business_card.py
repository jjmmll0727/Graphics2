import cv2
import numpy as np
from matplotlib import pyplot as plt
clicked = []

# 스페이스 바 대기
def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)

# 명함 바깥부분에 직선형태의 shape이 있어서 다 제거해주기 위해 grabcut을 함
def grabcut(resized):
    mask = np.zeros(resized.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (60, 60, 560, 460) #명함을 다 포함할수 있는 수평 수직의 직사각형의 좌상 우하 꼭짓점
    cv2.grabCut(resized, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    grabcutted = resized * mask2[:, :, np.newaxis]

    #cv2.imshow('grabcut', grabcutted)
    return grabcutted

def canny(grabcutted):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) ## 모폴로지 연산을 위한 커널
                                                               # 모폴로지 연산 이유: 배경과 전경의 밝기 차이가 적어 grabcut 함수를 사용하고도 굴곡을 지울수 없었다.
                                                               # 최대한 굴곡을 지우기 위해 모폴로지 연산을 수행함.
    gray = cv2.cvtColor(grabcutted, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0) # 한번더 굴곡을 줄이기 위해서 블러 처리함

    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=10)  ## 모폴로지 연산 closing 연산횟수: 10 그 이상 그 이하로 하면 형태가 뭉개져
    img_canny = cv2.Canny(closing, 5000, 1500, apertureSize = 5, L2gradient = True)
    cv2.imshow('canny', img_canny)
    return img_canny
    # canny 연산을 할때 이진 모폴로지와 가우시안 스무딩 까지 해서 명함안의 잡음 제거
'''
def drawlingline(approx):
    src = cv2.imread('card1.jpg')
    resized = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    # src = grabcut(resized)
    dst = resized.copy()

    cv2.drawContours(dst, [approx], -1, (0, 255, 0), 10)
    cv2.imshow('Outline', dst)
'''
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

    src = cv2.imread('card1.jpg')

    resized = cv2.resize(src, dsize=(640,480), interpolation=cv2.INTER_AREA)
    #src = grabcut(resized)
    dst = resized.copy()
    #screenCnt = None
    ( cnts, _) = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    cv2.drawContours(img_canny, [cnt], 0, (255, 255, 0), 1)
    epsilon = 0.03*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True) # 들어가는 순서가 자기마음대로

    size = len(approx) # 4변의 사각형임을 알수 있어

    cv2.line(dst, tuple(approx[0][0]), tuple(approx[size-1][0]), (0,255,0), 3)

    cv2.line(dst, tuple(approx[0][0]), tuple(approx[1][0]), (0, 255, 0), 3) # topleft부터 대입해서 approx[0][0] --> topleft
    cv2.line(dst, tuple(approx[1][0]), tuple(approx[2][0]), (0, 0, 255), 3)
    cv2.line(dst, tuple(approx[2][0]), tuple(approx[3][0]), (255, 0, 0), 3)
    cv2.line(dst, tuple(approx[3][0]), tuple(approx[0][0]), (255, 255, 0), 3)

    approx = approx.reshape(4,2)
    (topLeft, topRight, bottomRight, bottomLeft) = approx

    w1 = abs(bottomRight[0] - topRight[0])
    w2 = abs(topLeft[0] - bottomLeft[0])
    h1 = abs(topLeft[1] - topRight[1])
    h2 = abs(bottomLeft[1] - bottomRight[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    #print(maxHeight)
    #print(maxWidth)

    # 투영변횐
    src = []

    # 4좌표를 sorting 하고 4좌표끼리의 대소비교를 하여 src에 대입한다. jpg 파일마다 순서가 바뀌므로 sorting해서 일반화를 해야해
    src.append([float(bottomRight[0]), float(bottomRight[1])]) # 좌상 부터 시계방향으로 대입
    src.append([float(topRight[0]), float(topRight[1])])
    src.append([float(topLeft[0]), float(topLeft[1])])
    src.append([float(bottomLeft[0]), float(bottomLeft[1])])

    src.sort(key = lambda x:x[0]) #이차원배열을 x성분을 기준으로 sort
    src_tmp = [] # 좌상, 좌하, 우상, 우하 순서대로 sorting 해서 저장

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
    cv2.circle(dst, (int(src_tmp[3][0]), int(src_tmp[3][1])), 4, (230, 220, 0), -1)

    # 4좌표값 출력
    print(src_tmp[0])
    print(src_tmp[1])
    print(src_tmp[2])
    print(src_tmp[3])

    src_np = np.array(src_tmp, dtype=np.float32)

    # 좌상 좌하 우상 우하 순서대로 mapping
    dst_np = np.array([
        [0, 0],
        [0, maxHeight],
        [maxWidth, 0],
        [maxWidth, maxHeight]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src=src_np,
                                    dst=dst_np)  # 영상의 자동차 번호판을 template에 맞추는 투영행렬(M, 변환행렬) 을 구해준다. 직선 의 성질은 유지하되, 평행은 유지 x
    perspective = cv2.warpPerspective(resized, M=M, dsize=(maxWidth, maxHeight))  # 투영행렬(변환행렬)에 의해 결과물 출력

    ##resized = cv2.resize(result, dsize=(600, 300), interpolation=cv2.INTER_AREA)
    cv2.imshow('perspective', perspective)

    cv2.imshow('drawlingline', dst)


    '''
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 + peri, True)

        if len(approx) == 4:
            return approx
            break
            #ccopy(approx)
            #print(screenCnt)

    '''


    #lines = cv2.HoughLinesP(img_canny, 0.6, np.pi / 180, 45, minLineLength=15, maxLineGap=500)
    #lines = cv2.HoughLines(img_canny, 0.8, np.pi / 180, 45, 0, 0, min_theta=0, max_theta=np.pi)
    # cv2.HoughLinesP(검출 이미지, 거리, 각도, 임곗값, 최소 선 길이, 최대 선 간격)
    # 임계값 조절이 중요해 배경과 전경의 밝기 차이가 작기때문에
    # 최소 선 길이: 검출된 직선이 가져야 하는 최소한의 선 길이
    # 최대 선 간격: 검출된 직선들 사이의 최대 허용 간격
    '''
    count = 0;
    for i in lines:
        cv2.line(dst, (i[0][0], i[0][1]), (i[0][2], i[0][3]), (0, 0, 255), 1)
        ##count = count + 1
        ##if count == 6:
          ##  cv2.circle(dst, (i[0][0], i[0][1]),5, (255,0,0), -1)
           ## cv2.circle(dst, (i[0][2], i[0][3]), 5, (0, 255, 0), -1)
        print((i[0][0], i[0][1]), (i[0][2], i[0][3]) )
        print( float( (i[0][3] - i[0][1]) / (i[0][2] - i[0][0]) ) )
        print( (float( (i[0][3] - i[0][1]) / (i[0][2] - i[0][0]) )*(-i[0][0]) )+ i[0][1])
        print()
        ## 같은 집단에서 선분의 길이가 가장 긴 것들을 뽑으면 4개의 선분이된다.
        ## 기울기를 판단해서 어떠한 기준을 잡아야할듯
        ## 아니면 그냥 내맘대로 정하던가 
    '''
    '''
    count = 0
    for i in lines:
        count = count + 1
        rho, theta = i[0][0], i[0][1]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho

        scale = dst.shape[0] + dst.shape[1]

        x1 = int(x0 + scale * -b)
        y1 = int(y0 + scale * a)
        x2 = int(x0 - scale * -b)
        y2 = int(y0 - scale * a)
        if count < 5:
            cv2.line(dst, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(dst, (x0, y0), 3, (255, 0, 0), 5, cv2.FILLED)

    cv2.imshow('findline', dst)
    '''

def main():
    ori_img = cv2.imread('card1.jpg')
    resized = cv2.resize(ori_img, dsize=(640, 480), interpolation=cv2.INTER_AREA) # 영상의 사이즈가 너무 커서 조정함

    cv2.imshow('resize', resized)
    wait()
    canny(grabcut(resized))
    wait()
    drawlingCorrectly(canny(grabcut(resized)))
    wait()


if __name__ == "__main__":
    main()

