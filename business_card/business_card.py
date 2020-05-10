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

# 점진성 확률적 허프 변환
def findingLine(img_canny):
    src = cv2.imread('card2.jpg')
    resized = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    #src = grabcut(resized)
    dst = resized.copy()

    lines = cv2.HoughLinesP(img_canny, 0.6, np.pi / 180, 45, minLineLength=15, maxLineGap=500)
    # cv2.HoughLinesP(검출 이미지, 거리, 각도, 임곗값, 최소 선 길이, 최대 선 간격)
    # 임계값 조절이 중요해 배경과 전경의 밝기 차이가 작기때문에
    # 최소 선 길이: 검출된 직선이 가져야 하는 최소한의 선 길이
    # 최대 선 간격: 검출된 직선들 사이의 최대 허용 간격
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

    cv2.imshow('findline', dst)



def main():
    ori_img = cv2.imread('card2.jpg')
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


if __name__ == "__main__":
    main()

