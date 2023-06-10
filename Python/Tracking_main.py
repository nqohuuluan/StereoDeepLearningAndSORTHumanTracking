import cv2
import time
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import numpy as np
import calibration
from cvzone.PoseModule import PoseDetector
from PIL import Image
import triangulation as tri

detector = PoseDetector()

# Stereo vision setup parameters
B = 18             #Distance between the cameras [cm]
f = 2.8              #Camera lense's focal length [mm]
alpha = 30           #Camera field of view in the horisontal plane [degrees]

# Quet nhung khung hinh dau tien de tim bbox
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detectorR = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
detectorL = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

trackerR = cv2.legacy.TrackerCSRT_create()
trackerL = cv2.legacy.TrackerCSRT_create()

capR = cv2.VideoCapture('LL_rz.mp4')
capL = cv2.VideoCapture('LR_rz.mp4')
# capR = cv2.VideoCapture(3)
# capL = cv2.VideoCapture(1)
rects_right = []
rects_left = []
start_time = time.time()
while time.time() - start_time < 5:
    successR, imgR = capR.read()
    successL, imgL = capL.read()

    # imgR = cv2.resize(imgR, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    # imgL = cv2.resize(imgL, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    # imgR = imutils.resize(imgR, width = 640, height = 360)
    # imgL = imutils.resize(imgL, width = 640, height = 360)

    (HR, WR) = imgR.shape[:2]  # shape/2 là lấy hai phần tử đầu tiên của mảng frame
    (HL, WL) = imgL.shape[:2]

    # blob_right = cv2.dnn.blobFromImage(imgR, 0.007843, (WR, HR), 127.5)
    # blob_left = cv2.dnn.blobFromImage(imgL, 0.007843, (WL, HL), 127.5)

    blob_right = cv2.dnn.blobFromImage(imgR, 0.007843, (WR, HR), 127.5)
    blob_left = cv2.dnn.blobFromImage(imgL, 0.007843, (WL, HL), 127.5)

    detectorR.setInput(blob_right, "data")
    person_detections_right = detectorR.forward()

    detectorL.setInput(blob_left, "data")
    person_detections_left = detectorL.forward()


    global bboxR, bboxL

    for iR, iL in zip(np.arange(0, person_detections_right.shape[2]), np.arange(0, person_detections_left.shape[2])):
        confidenceR = person_detections_right[0, 0, iR, 2]
        confidenceL = person_detections_left[0, 0, iL, 2]

        if confidenceR > 0.5:
            idxR = int(person_detections_right[0, 0, iR, 1])

            if CLASSES[idxR] != "person":
                continue

            person_box_right = person_detections_right[0, 0, iR, 3:7] * np.array([WR, HR, WR, HR])
            person_box_right = person_box_right.astype(int)
            rects_right.append(person_box_right)

        if confidenceL > 0.5:
            idxL = int(person_detections_left[0, 0, iL, 1])

            if CLASSES[idxL] != "person":
                continue

            person_box_left = person_detections_left[0, 0, iL, 3:7] * np.array([WL, HL, WL, HL])
            person_box_left = person_box_left.astype(int)
            rects_left.append(person_box_left)
            # if time.time() - start_time >= 10:
            #     break

print("rectR", rects_right)
print("rectL", rects_left)
print("-------------")
bboxR2D = np.array(rects_right)
bboxL2D = np.array(rects_left)

DL_bboxR = bboxR2D[-1]
DL_bboxL = bboxL2D[-1]
print(type(DL_bboxR))

adjR = [-100, 0, 100, 0]

DL_bboxR = DL_bboxR+adjR
DL_bboxL = DL_bboxL+adjR

xR, yR, wR, hR = DL_bboxR[0], DL_bboxR[1], DL_bboxR[2]-DL_bboxR[0], DL_bboxR[3]-DL_bboxR[1]
xL, yL, wL, hL = DL_bboxL[0], DL_bboxL[1], DL_bboxL[2]-DL_bboxL[0], DL_bboxL[3]-DL_bboxL[1]

bboxR = xR, yR, wR, hR
bboxL = xL, yL, wL, hL

# bboxR = cv2.selectROI("TrackingR", imgR, False)
# bboxL = cv2.selectROI("TrackingL", imgL, False)

trackerR.init(imgR, bboxR)
trackerL.init(imgL, bboxL)

def drawBoxR(imgR, bboxR):
    xR, yR, wR, hR = int(bboxR[0]), int(bboxR[1]), int(bboxR[2]), int(bboxR[3])
    cv2.rectangle(imgR, (xR,yR), ((xR+wR),(yR+hR)),(255,0,255),3,1)

def drawBoxL(imgL, bboxL):
    xL, yL, wL, hL = int(bboxL[0]), int(bboxL[1]), int(bboxL[2]), int(bboxL[3])
    cv2.rectangle(imgL, (xL, yL), ((xL + wL), (yL + hL)), (255, 0, 255), 3, 1)

prev_time = 0
while True:
    successR, imgR = capR.read()
    successL, imgL = capL.read()

    # imgR, imgL = calibration.undistortRectify(imgR, imgL)

    successR, bboxR = trackerR.update(imgR)
    successL, bboxL = trackerL.update(imgL)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    cv2.putText(imgR, "FPS: "+str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(imgL, "FPS: " + str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    prev_time = current_time

    global cxR, cyR, cxL, cyL, centerR, centerL
    if successR:
        drawBoxR(imgR, bboxR)
        # posR = trackerR.pos
        for i in bboxR:
            xR, yR, wR, hR = bboxR
            bboxR = np.array(bboxR)
            bboxR = bboxR.astype(int)
            maskR = np.zeros(imgR.shape[:2], dtype=np.uint8)
            maskR[bboxR[1]:bboxR[1]+bboxR[3], bboxR[0]:bboxR[0]+(bboxR[2])] = 255
            mask_imgR = cv2.bitwise_and(imgR, imgR, mask=maskR)
            mask_imgR = detector.findPose(mask_imgR)
            # lmList tra ve 3 lan gia tri moi mark
            lmListR, bboxInfo = detector.findPosition(mask_imgR, draw= False, bboxWithHands=True)
            if len(lmListR)>0:
                markR_arr_id11 = lmListR[11]
                markR_arr_id12 = lmListR[12]
                centerR = int(markR_arr_id11[1] + (markR_arr_id12[1] - markR_arr_id11[1])/2), int(markR_arr_id11[2] + (markR_arr_id12[2] - markR_arr_id11[2])/2)
                # print("centerR: ", type(centerR))
    else:
        cv2.putText(imgR, "Object can not be tracked!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if successL:
        drawBoxL(imgL, bboxL)
        # global cxL, cyL
        for i in bboxL:
            xL, yL, wL, hL = bboxL
            bboxL = np.array(bboxL)
            bboxL = bboxL.astype(int)
            maskL = np.zeros(imgL.shape[:2], dtype=np.uint8)
            maskL[bboxL[1]:bboxL[1] + bboxL[3], bboxL[0]:bboxL[0] + bboxL[2]] = 255
            mask_imgL = cv2.bitwise_and(imgL, imgL, mask=maskL)
            mask_imgL = detector.findPose(mask_imgL)
            lmListL, bboxInfo = detector.findPosition(mask_imgL, draw= False, bboxWithHands=True)
            if len(lmListL)>0:
                markL_arr_id11 = lmListL[11]
                markL_arr_id12 = lmListL[12]
                centerL = int(markL_arr_id11[1] + (markL_arr_id12[1] - markL_arr_id11[1]) / 2), int(markL_arr_id11[2] + \
                          (markL_arr_id12[2] - markL_arr_id11[2]) / 2)
                # print("centerL: ", centerL)
    else:
        cv2.putText(imgR, "Object can not be tracked!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(imgL, "Object can not be tracked!", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    depth = tri.find_depth(centerR, centerL, imgR, imgL, B, f, alpha)
    cv2.putText(imgR, "DistanceR: " + str(round(depth, 1)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)
    cv2.putText(imgL, "DistanceL: " + str(round(depth, 1)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)
    cv2.circle(mask_imgR, centerR, 5, (255, 0, 0), cv2.FILLED)
    cv2.circle(mask_imgL, centerL, 5, (255, 0, 0), cv2.FILLED)
    print("depth: ", int(depth))
    # imgR = cv2.resize(imgR, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    # imgL = cv2.resize(imgL, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_LINEAR)
    # imgR = imutils.resize(imgR, width=640, height=360)
    # imgL = imutils.resize(imgL, width=640, height=360)
    stacked_frames = np.hstack((imgR, imgL))
    stacked_frames2 = np.hstack((mask_imgR, mask_imgL))
    cv2.imshow("frame", stacked_frames)
    cv2.imshow('Masked Image', stacked_frames2)

    if cv2.waitKey(1) & 0XFF ==ord('q'):
        break
    # time.sleep(1)
capR.release()
capL.release()
cv2.destroyAllWindows()