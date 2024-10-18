import math
import cv2
import cvzone
import pickle
import numpy as np
from ultralytics import YOLO
import serial
import time

# --------------- Variables -------------------
p1 = 1
roadCam = 0
x, y = 400, 400
color = (0, 200, 0)
count = 50
avanow1 = 6
sign = " WAIT ! "

polyroad = 'Road1C.p'
obj_road = open(polyroad, 'rb')
roi_road = pickle.load(obj_road)
obj_road.close()

statusPalang1 = False
# assign status palang parkiran

polygonKeluar1 = 'parkiran_out1C.p'
# assign semua polygons Keluar parkiran

polygonParkir1 = 'parkiran1C.p'
# assign semua polygons parkiran

model_path = r"D:\YOLO\YOLO\pythonProject\model\model very accurate\best.pt"


confidence = 0.75

class_names = ["SEDAN", "SUV", "TRUCK"]

file_obj_parkir1 = open(polygonParkir1, 'rb')
roisParkir1 = pickle.load(file_obj_parkir1)

file_obj_keluar1 = open(polygonKeluar1, 'rb')
roisKeluar1 = pickle.load(file_obj_keluar1)

file_obj_keluar1.close()

file_obj_parkir1.close()

arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)  # connect to arduino

cam_width, cam_height = 1280, 720  # CAM RES

cap1 = cv2.VideoCapture(p1)  # WEB CAM ASSIGN
cap2 = cv2.VideoCapture(roadCam)

cap2.set(3, cam_width)
cap2.set(4, cam_height)# SET ALL FRAME SIZE
cap1.set(3, cam_width)
cap1.set(4, cam_height)



model = YOLO(model_path)  # SELECT MODEL USED

def count_Spaces(ava_spaces, _object_list, _parking_spaces):
    for parking_space in _parking_spaces:
        ret = 0
        empty = True

        # Convert polygon to numpy array and reshape
        parking_space = np.array(parking_space, np.int32).reshape((-1, 1, 2))

        # Check if any car is present in this polygon
        for obj in _object_list:
            car_center = obj["center"]
            result = cv2.pointPolygonTest(parking_space, car_center, False)
            if result > 0:
                empty = False
                ret += 1
                break
        if not empty:
            ava_spaces -= ret
    return ava_spaces

def overlay_polygons(_image, _object_list, _parking_spaces, _draw_occupied=False):
    overlay = _image.copy()
    global is_empty
    for parking_space in _parking_spaces:
        is_empty = True

        # Convert polygon to numpy array and reshape
        parking_space = np.array(parking_space, np.int32).reshape((-1, 1, 2))

        # Check if any car is present in this polygon
        for obj in _object_list:
            car_center = obj["center"]
            result = cv2.pointPolygonTest(parking_space, car_center, False)
            if result > 0:
                is_empty = False
                break

        if is_empty:
            cv2.fillPoly(overlay, [parking_space], (0, 255, 0))  # Green for empty space
        if not is_empty and _draw_occupied:
            cv2.fillPoly(overlay, [parking_space], (0, 0, 255))  # Red for occupied space

    cv2.addWeighted(overlay, 0.35, _image, 0.65, 0, _image)
    return is_empty

def get_object_list_yolo(_model, _img, _class_names, _confidence=0.5, draw=True):
    _results = _model(_img, stream=False, verbose=False)
    _object_list = []
    for r in _results:
        boxes = r.boxes
        for box in boxes:
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > _confidence:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                center = x1 + (w // 2), y1 + (h // 2)
                class_name = _class_names[int(box.cls[0])]

                _object_list.append({"bbox": (x1, y1, w, h),
                                     "center": center,
                                     "conf": conf,
                                     "class": class_name})

                if draw:
                    cvzone.cornerRect(_img, (x1, y1, w, h))
                    cvzone.putTextRect(_img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=2, thickness=1)
    return _object_list

def write(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)


cv2.namedWindow("Image1"), cv2.namedWindow("camRoad")  # NAMING ALL WINDOW TO MOVE IT
cv2.moveWindow("Image1", 0, 100), cv2.moveWindow("camRoad", 964, 0) #MOVE ALL WINDOWS TO CERTAIN COORDINATE


def frameDetectCar():
    global center, result
    global statusPalang1
    global color
    global count
    global avanow1
    global sign
    reset = 50
    park_spaces = 5

    success1, img1 = cap1.read()
    success2, img2 = cap2.read()  # READING CAMERA ASSIGNED



    object_list1 = get_object_list_yolo(model, img1, class_names, confidence, draw=True)
    object_list2 = get_object_list_yolo(model, img2, class_names, confidence, draw=True)  # CAMERA FRAME GET DETECTED WITH YOLO
    ava1 = count_Spaces(park_spaces, object_list1, roisParkir1)

    # ---------------------------------- PARKIRAN KELUAR --------------------------------------------------------------------------
    empty_status1 = overlay_polygons(img1, object_list1, roisKeluar1, _draw_occupied=True)
    # status polygons keluar

    keluar_space1 = np.array(roisKeluar1, np.int32).reshape(
        (-1, 1, 2))  # to detect center point polygon keluar(convert polygon to array)
    # -----------------------------------------------------------------------------------------------------------------------------

    if ((len(object_list1) != 0) and (empty_status1 == False)):
        # Check if any car is present in this polygon
        for obj in object_list1:
            car_center = obj["center"]
            result = cv2.pointPolygonTest(keluar_space1, car_center, False)
        if ((statusPalang1 == False) and (result > 0)):
            write("2")
            statusPalang1 = True
            avanow1 = ava1
        # elif ((statusPalang1 == True) and (result <= 0)):
        #     write("3")
    # ----------------------------------- PARKIRAN --------------------------------------------------------------------------------

    overlay_polygons(img1, object_list1, roisParkir1, _draw_occupied=True)  # Parkiran


    if ava1 == 0:
        color = (0, 0, 255)  # Red for no available spaces
    else:
        color = (0, 200, 0)  # Green for available spaces
    cvzone.putTextRect(img1, f"Available: {ava1}/{str(park_spaces)}", (20, 50), colorR=color)


    # -------------------------------- ROAD ---------------------------------------------------------------------------------------

    empty_status_road = overlay_polygons(img2, object_list2, roi_road, _draw_occupied=True)

    parking_space = np.array(roi_road, np.int32).reshape((-1, 1, 2))


    if ((len(object_list2) != 0) and (empty_status_road == False)):

        # Check if any car is present in this polygon
        for obj in object_list2:
            car_center = obj["center"]
            result = cv2.pointPolygonTest(parking_space, car_center, False)
        if ((statusPalang1 == False) and (result >0)):
            write("2")
            avanow1 = ava1
            statusPalang1 = True

# --------------------------------- CLOSE ALL PALANG WHEN CAR GOT IN ---------------------------------------------------------------
    if (statusPalang1 == True):
        if (ava1 < avanow1):
            write("3")
            statusPalang1 = False
            sign = "WAIT! ... "


#--------------------------------- CLOSE ALL PALANG AFTER SPECIFIC AMOUNT OF TIME --------------------------------------------------
    if(count == 0):
        if (statusPalang1 == True and result<=0):
            write("3")
            print("close")
            statusPalang1 = False
        count = reset
    count-=1
    print(count)


    #---------------------------------- CAM SHOW -----------------------------------------------------------------------------------
    cv2.imshow("Image1", cv2.resize(img1, (480, 272)))
    cv2.imshow("camRoad", cv2.resize(img2, (480, 272)))  # SHOW CAMERA FRAME AND RESIZE TO 480x272

    cv2.waitKey(1)


while True:
    frameDetectCar()
