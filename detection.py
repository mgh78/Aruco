import cv2
import cv2.aruco as aruco

#  Load image
inputImage = cv2.imread("/Users/mahdi/Desktop/object/marker49.png")

#  Define dictionary and detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# White border so markers generated with generateImageMarker (tight crop) still detect
h, w = inputImage.shape[:2]
pad = max(16, min(h, w) // 20)
inputImage = cv2.copyMakeBorder(
    inputImage, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255)
)

# Detect markers
markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(inputImage)

print("markerCorners:", markerCorners)
print("markerIds:", markerIds)
print("rejectedCandidates:", rejectedCandidates)    



# Draw detected markers
if markerIds is not None:
    cv2.aruco.drawDetectedMarkers(inputImage, markerCorners, markerIds)
    print("Detected IDs:", markerIds)
else:
    print("No markers detected")

# Show result
cv2.imshow("Detected Markers", inputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()