import sys

import cv2
import cv2.aruco as aruco

# Same dictionary / ID family as creation.py (marker49.png uses id 49).
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

if sys.platform == "darwin":
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    sys.exit(1)

print("Live ArUco test — DICT_4X4_50 (e.g. marker id 49 from creation.py). ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("creation2 — VideoCapture", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
