import cv2
import multiprocessing as mp
import numpy as np
import numpy as np
from cvzone.PoseModule import PoseDetector
import time
import triangulation as tri

# Set up
detector = PoseDetector()
# Stereo vision setup parameters
B = 18             #Distance between the cameras [cm]
f = 2.8              #Camera lense's focal length [mm]
alpha = 30
# Quet nhung khung hinh dau tien de tim bbox
protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
DL_detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
start_time = time.time()

video_path1 = 'LR_rz.mp4'
video_path2 = 'LL_rz.mp4'
# video_path1 = 3
# video_path2 = 1
# Khởi tạo CSRT Tracker
tracker = cv2.legacy.TrackerCSRT_create()

def track_object(video_path,output):


    # Đọc video đầu vào
    cap = cv2.VideoCapture(video_path)
    rects = []
    start_time = time.time()
    while time.time() - start_time < 5:
        success, frame = cap.read()

        (H, W) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        DL_detector.setInput(blob, "data")
        person_detections = DL_detector.forward()

        # global bbox
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idxR = int(person_detections[0, 0, i, 1])
                if CLASSES[idxR] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                person_box = person_box.astype(int)
                rects.append(person_box)
    bbox2D = np.array(rects)
    DL_bbox = bbox2D[-1]
    adj = [-50, 0, 50, 0]
    DL_bbox = DL_bbox + adj
    x, y, w, h = DL_bbox[0], DL_bbox[1], DL_bbox[2] - DL_bbox[0], DL_bbox[3] - DL_bbox[1]
    bbox = x, y, w, h
    tracker.init(frame, bbox)
    # Khởi tạo CSRT Tracker
    while True:
        # Đặt vị trí ban đầu của đối tượng cần tracking
        success, frame = cap.read()
        # Tracking đối tượng trong frame
        success, bbox = tracker.update(frame)
        if success:
        # Nếu tracking thành công, vẽ bbox lên frame 1
            x, y, w, h = [int(v) for v in bbox]
            # print(x,y,w,h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for i in bbox:
                x, y, w, h = bbox
                bbox = np.array(bbox)
                bbox = bbox.astype(int)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+(bbox[2])] = 255
                mask_img = cv2.bitwise_and(frame, frame, mask=mask)
                mask_img = detector.findPose(mask_img)
                # lmList tra ve 3 lan gia tri moi mark
                lmList, bboxInfo = detector.findPosition(mask_img, draw= False, bboxWithHands=True)
                if len(lmList)>0:
                    mark_arr_id11 = lmList[11]
                    mark_arr_id12 = lmList[12]
                    center = int(mark_arr_id11[1] + (mark_arr_id12[1] - mark_arr_id11[1])/2), int(mark_arr_id11[2] + (mark_arr_id12[2] - mark_arr_id11[2])/2)
                    output.put((center, frame))
        else:
            output.put(None)
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
        # Hiển thị frame 1 và frame 2
        cv2.circle(frame, center, 5, (255, 0, 0), cv2.FILLED)
        cv2.imshow('Video 1', frame)
        # cv2.imshow('Video 2', frame2)
    # Giải phóng bộ nhớ và đóng video
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Đường dẫn đến hai video cần tracking
    video1_path = '../AIComputerVision-master/video/LL_rz.mp4'
    video2_path = '../AIComputerVision-master/video/LR_rz.mp4'

    # Đọc frame đầu tiên của hai video
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # Vị trí ban đầu của đối tượng cần tracking trong hai video
    # bbox1 = (100, 100, 200, 200)  # (x, y, w, h)
    # bbox2 = (50, 50, 150, 150)  # (x, y, w, h)
    # bbox1 = cv2.selectROI("TrackingR", frame1, False)
    # bbox2 = cv2.selectROI("TrackingL", frame2, False)
    # while True:
    #     success1, frame1 = cap1.read()
    #     success2, frame2 = cap2.read()

    # Tạo hai queue để lưu bbox của đối tượng trong hai video
    queue1 = mp.Queue()
    queue2 = mp.Queue()

    # Tạo hai tiến trình để tracking đối tượng trong hai video
    p1 = mp.Process(target=track_object, args=(video_path1, queue1))
    p2 = mp.Process(target=track_object, args=(video_path2, queue2))
    # Khởi động hai tiến trình
    p1.start()
    p2.start()
    output1 = []
    output2 = []
    while True:
        # ret1, frame1 = cap1.read()
        # ret2, frame2 = cap2.read()

        # Lấy bbox của đối tượng từ queue
        result1 = queue1.get()
        result2 = queue2.get()
        center1, frame1 = result1
        center2, frame2 = result2

        output1.append((center1, frame1))
        output2.append((center2, frame2))
        # print(output1[0])
        for (center1, frame1), (center2, frame2) in zip(output1, output2):
            print("center1: ", center1)
            print("center2: ", center2)
            print("---------")
            # cv2.imshow("test", frame1)
            # print("center2: ", center2)
            depth = tri.find_depth(center1, center2, frame1, frame2, B, f, alpha)
            print("depth: ", depth)
        # Đợi cho hai tiến trình kết thúc
    p1.join()
    p2.join()