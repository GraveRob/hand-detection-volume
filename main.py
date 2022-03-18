import cv2;
import mediapipe as mp;
import time;
import math;
import numpy as np;
from ctypes import cast, POINTER;
from comtypes import CLSCTX_ALL;
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume;

mp_hands = mp.solutions.hands;
mp_drawing = mp.solutions.drawing_utils;

devices = AudioUtilities.GetSpeakers();
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None);
volume = cast(interface, POINTER(IAudioEndpointVolume));
volRange = volume.GetVolumeRange();
volume.SetMasterVolumeLevel(0, None);
minVal = volRange[0];
maxVal = volRange[1];

widthCam, heightCam = 800, 700;

cap = cv2.VideoCapture(0);
cap.set(3, widthCam);
cap.set(4, heightCam);

lasTime = 0;

with mp_hands.Hands(
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:
    while cap.isOpened():
        success, img = cap.read();

        start = time.time();

        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB);
        img.flags.writeable = False;
        detections = hands.process(img);
        img.flags.writeable = True;
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR);

        lmlist = [];

        if detections.multi_hand_landmarks:
            for hand_landmarks in detections.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS);
            
            for id, lm in enumerate(detections.multi_hand_landmarks[0].landmark):
                h, w, c = img.shape;
                x, y = int(lm.x * w), int(lm.y * h);
                lmlist.append([x, y]);

            if len (lmlist):
                x1, y1 = lmlist[4][0], lmlist[4][1];
                x2, y2 = lmlist[8][0], lmlist[8][1];

                length = math.hypot(x2 - x1, y2 - y1);

                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED);
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED);

                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.FILLED);

                color =  (0, 0, 255) if length < 20 else (255, 0, 0) if length > 150 else (0, 255, 0); 
                cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 8, color, cv2.FILLED);

                vol = np.interp(length, [50, 300], [minVal, maxVal]);
                volume.SetMasterVolumeLevel(vol, None);

        end = time.time();
        fps = 1 / (end - start);
        cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3);

        cv2.imshow('RESULT', img);

        if cv2.waitKey(5) & 0xFF == 27:
            break;
cap.release();