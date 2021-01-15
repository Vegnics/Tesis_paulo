import cv2
import numpy as np

def click_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("X= {}, Y={}".format(x,y)) #
    return

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_rgb)
image_folder="/home/amaranth/Desktop/TESIS/Tesis_paulo/calibration_images/new_calibration_images/"
image=cv2.imread(image_folder+"extrinsic_calibration2.jpg")

gray_img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
retval, corners = cv2.findChessboardCorners(gray_img, patternSize=(7,7))

#inicializamos el primer punto espacial
p_0_x=510.2+6*31-610
p_0_y=445.2+6*31-482
z=-537.5+99.96
Ap=np.zeros((1,49,3),dtype=np.float32)

intrinsics=np.load("intrinsics.npy")
distcoeffs=np.load("distcoeffs.npy")

for j in range(7):
    for i in range(7):
        Ap[0,i+7*j,:3]=[p_0_x-j*31,p_0_y-i*31,z]

print(Ap)

drawed=cv2.drawChessboardCorners(image,(7,7),corners,retval)
cv2.imshow("image",drawed)
cv2.waitKey(0)
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([Ap], [corners], gray_img.shape[::-1],cameraMatrix=intrinsics,distCoeffs=distcoeffs,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
rmatrix=cv2.Rodrigues(rvecs[0])
print(rmatrix)

np.save("rmatrix.npy",rmatrix[0])
np.save("tvec.npy",tvecs[0])






