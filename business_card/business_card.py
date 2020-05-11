import cv2
import numpy as np
from matplotlib import pyplot as plt
clicked = []

def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)

def grabcut(resized):
    mask = np.zeros(resized.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (60, 60, 560, 460)
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
    gray = cv2.GaussianBlur(gray, (3,3), 0) # 굴곡을 줄이기 위해서 블러 처리함

    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=10)  ## 모폴로지 연산 closing 연산횟수: 10
    img_canny = cv2.Canny(closing, 5000, 1500, apertureSize = 5, L2gradient = True)
    cv2.imshow('canny', img_canny)
    return img_canny

def drawlingLine(approx):
    src = cv2.imread('card2.jpg')
    resized = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    # src = grabcut(resized)
    dst = resized.copy()

    cv2.drawContours(dst, [approx], -1, (0, 255, 0), 10)
    cv2.imshow('Outline', dst)

def order_points(pts):
    rect = np.zeros((4,2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmin(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmin(diff)]

    return rect

# 점진성 확률적 허프 변환
def findingLine(img_canny):

    # 여기서 부터 네 선분 그리기
    src = cv2.imread('card1.jpg')

    #r = 800.0/src.shape[0]
    #dim = (int(src.shape[1] * r), 800)
    resized = cv2.resize(src, dsize=(640,480), interpolation=cv2.INTER_AREA)
    #src = grabcut(resized)
    dst = resized.copy()
    #screenCnt = None
    ( cnts, _) = cv2.findContours(img_canny.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnts[0]
    cv2.drawContours(img_canny, [cnt], 0, (255, 255, 0), 1)
    epsilon = 0.03*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(dst, [approx], -1, (0,255,0), 3) # approx : 초록색 사각형 객체

    # 여기서 부터 꼭짓점 찍기
    #print(approx)
    #rect = order_points(approx.reshape(4, 2))
    #print(rect)
    approx = approx.reshape(4,2)
    (topLeft, topRight, bottomRight, bottomLeft) = approx

    print(topLeft[0], topLeft[1]) #좌상
    print(topRight[0], topRight[1])  # 좌하
    print(bottomLeft[0], bottomLeft[1]) #우상
    print(bottomRight[0], bottomRight[1]) #우하


    cv2.circle(dst, (int(bottomLeft[0]), int(bottomLeft[1])), 4, (255, 0, 0), -1)
    cv2.circle(dst, (int(bottomRight[0]), int(bottomRight[1])), 4, (0, 0, 255), -1)
    cv2.circle(dst, (int(topLeft[0]), int(topLeft[1])), 4, (0, 255, 255), -1)
    cv2.circle(dst, (int(topRight[0]), int(topRight[1])), 4, (230, 220, 0), -1)

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])

    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])


    cv2.imshow('tttt', dst)


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
    '''
    grabcut(resized)
    wait()
    '''
    canny(grabcut(resized))
    #wait()
    findingLine(canny(grabcut(resized)))
    wait()
    #drawlingLine(findingLine(canny(grabcut(resized))))
    #wait()



if __name__ == "__main__":
    main()

