import cv2
import numpy as np
def click_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X= {}, Y={}".format(x,y)) #
    return

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_rgb)

image_num=10
image_folder="/home/amaranth/Desktop/TESIS/Tesis_paulo/calibration_images/new_calibration_images/"
images=[]

objpoints = np.zeros((1,7*7,3), np.float32)

for x in range(7):
    for y in range(7):
        objpoints[0,y+7*x,:3]=[x*31,y*31,0]

for i in range(10):
    image=cv2.imread(image_folder+"chess_{x}.jpg".format(x=i+1),1)
    images.append(image)

total_corners=[]
print(objpoints)

for img in images:
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    retval, corners = cv2.findChessboardCorners(gray_img, patternSize=(7,7))
    total_corners.append(corners)
    drawed=cv2.drawChessboardCorners(img,(7,7),corners,retval)
    cv2.imshow("image",drawed)
    cv2.waitKey(0)
cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints,objpoints,objpoints,objpoints,objpoints,objpoints,objpoints,objpoints,objpoints,objpoints], total_corners, images[0].shape[::-1][1::],None,None)
#np.save("intrinsics.npy",mtx)
#np.save("distcoeffs.npy",dist )
print(mtx)