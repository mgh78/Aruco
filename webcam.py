import cv2
import cv2.aruco as aruco
import numpy as np

# Open camera (Mac)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

# ArUco setup
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Camera parameters (approximate)
camera_matrix = np.array([
    [600, 0, 320],
    [0, 600, 240],
    [0, 0, 1]
], dtype=float)

dist_coeffs = np.zeros((5, 1))

# Marker size (IMPORTANT: meters!)
marker_length = 0.05  # 5 cm

# Object points (corner-based coordinate system)
_obj_pts = np.array([
    [0, 0, 0],
    [marker_length, 0, 0],
    [marker_length, marker_length, 0],
    [0, marker_length, 0]
], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame from camera.")
        break

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None and len(corners) > 0:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            img_pts = corners[i][0].astype(np.float32)

            ok, rvec, tvec = cv2.solvePnP(
                _obj_pts, img_pts, camera_matrix, dist_coeffs
            )

            if ok:
                # Draw axis only
                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03
                )

                # Print in terminal
                print("----- Marker -----")
                print(f"ID: {ids[i][0]}")
                print(f"tvec: x={tvec[0][0]:.3f}, y={tvec[1][0]:.3f}, z={tvec[2][0]:.3f}")
                print(f"rvec: x={rvec[0][0]:.3f}, y={rvec[1][0]:.3f}, z={rvec[2][0]:.3f}")
                print()

    cv2.imshow("Frame", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()