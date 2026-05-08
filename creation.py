import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker = aruco.generateImageMarker(aruco_dict, 49, 300)
# Quiet zone: detector needs white margin around the pattern (tight crops often fail).
pad = max(16, marker.shape[0] // 20)
marker = cv2.copyMakeBorder(
    marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255
)

cv2.imwrite("marker49.png", marker)