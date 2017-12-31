import cv2 as cv
import numpy as np
import math
import os
import matplotlib.pyplot as plt

cam = cv.VideoCapture("Dystonia, Head Tremors, and Depression.mp4")
drawBackground = cv.imread("dotBackground.png")
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
dot = list()

def distance(x1, y1, x2, y2):
    y = y2-y1
    x = x2-x1
    return int(math.sqrt(math.pow(x, 2) + math.pow(y, 2)))

def compareImage(img1, img2):
    orb = cv.ORB_create()
    img3 = None

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:25], img3, flags=0)

    plt.imshow(img3)
    plt.show()

def compareImage2(compare, cryto):
    compare = cv.cvtColor(compare, cv.COLOR_BGR2GRAY)
    cryto = cv.cvtColor(cryto, cv.COLOR_BGR2GRAY)

    _, thr1 = cv.threshold(compare, 127, 255, 0)
    _, cnt1, _ = cv.findContours(thr1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(cnt1)

    _, thr2 = cv.threshold(cryto, 127, 255, 0)
    _, cnt2, _ = cv.findContours(thr2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(cnt2)

    cv.drawContours(compare, cnt1, 1, (0, 255, 255), 2)
    cv.drawContours(cryto, cnt2, 1, (0, 255, 255), 2)

    cv.imshow('hihi', compare)
    cv.imshow('hihi2', cryto)

    cv.waitKey(0)

    result = cv.matchShapes(cnt1[0], cnt2[0], 1, 0.0)
    #print(result)

def extractCompareRegion(img):
    returnROI = None
    tmp = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_canny = cv.Canny(tmp, 100, 255)

    aaa = img_canny
    _, contours, _ = cv.findContours(aaa, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        maxIdx = 0
        maxArea = cv.contourArea(contours[0])
        for i in range(1, len(contours)):
            if cv.contourArea(contours[i]) > maxArea: maxIdx = i

        x, y, w, h = cv.boundingRect(contours[maxIdx])

        returnROI = img[y:y+h, x:x+w]
        cv.imshow('aaa', returnROI)
        cv.waitKey(0)
        return returnROI
    else:
        print("영역이 없습니다.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        circle_x = int(x+(w/2))
        circle_y = int(y+(h/2))
        facePoint = [circle_x, circle_y]

        if(len(dot)):
            prePoint = dot[len(dot)-1]
            pre_nowD = distance(prePoint[0], prePoint[1], facePoint[0], facePoint[1])
            if pre_nowD > 15 : dot.append(facePoint) # 두 점 사이의 거리가 너무 가까우면 점 목록에 등록하지 않음
            print("Distance", pre_nowD)

            tmp = np.array(dot) # numpy 배열로 등록 이유 -> 함수 내에서 numpy 형 배열을 사용
            cv.polylines(drawBackground, np.int32([tmp]), 1, (255, 255, 0), 2) # 점들을 이용해서 비교 사진에 polyline 들을 그린다.
            cv.polylines(frame, np.int32([tmp]), 1, (255, 255, 0), 2) # 어떻게 그려주는지 보여주기 위함

        else:
            dot.append(facePoint)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
    cv.imshow('img', frame)

    if cv.waitKey(30) & 0xFF == ord('q'):
        if os.path.exists("crypto.jpg"): # 암호파일이 존재하면
            crypto = cv.imread("crypto.jpg")
            drawBackground = extractCompareRegion(drawBackground)
            #compareImage2(drawBackground, crypto)

        else:
            drawBackground = extractCompareRegion(drawBackground)
            cv.imwrite("crypto.jpg", drawBackground)
            print("암호가 새로 저장되었습니다.")
        break

cv.destroyAllWindows()
quit(0)