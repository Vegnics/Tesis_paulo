import cv2
import numpy as np

def click_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        X=abs(z)*(x-ppx)/fx
        Y=abs(z)*(y-ppy)/fy
        Z=abs(z)
        P=np.array([X,Y,Z])
        t=np.resize(np.array([tx,ty,tz]),(3,1))
        P = np.resize(P, (3, 1))
        P[0] = P[0] - tx
        P[1] = P[1] - ty
        P[2] = P[2]
        P=np.matmul(rinvmat,P)
        print(rmatrix)
        print("X= {}, Y={}, Z={}".format(P[0],P[1],P[2])) #
        print("\n")
    return

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_rgb)
image_folder="/home/amaranth/Desktop/TESIS/Tesis_paulo/calibration_images/new_calibration_images/"
image=cv2.imread(image_folder+"extrinsic_calibration2.jpg")
mapx=np.load("mapx.npy")
mapy=np.load("mapy.npy")

nimage = cv2.remap(image,mapx,mapy,interpolation=cv2.INTER_LINEAR)

gray_img=cv2.cvtColor(nimage,cv2.COLOR_BGR2GRAY)
retval, corners = cv2.findChessboardCorners(gray_img, patternSize=(7,7))

#inicializamos el primer punto espacial
p_0_x=510.2+6*31-610
p_0_y=445.2+6*31-482
z=-537.5+99.96
Ap=np.zeros((1,49,3),dtype=np.float32)

print([p_0_x,p_0_y])

intrinsics=np.load("nintrinsics.npy")
distcoeffs=np.load("distcoeffs.npy")
rmatrix=np.load("rmatrix.npy")
tvec=np.load("tvec.npy")
ppx=intrinsics[0,2]
ppy=intrinsics[1,2]
fx=intrinsics[0,0]
fy=intrinsics[1,1]

rinvmat=np.linalg.inv(rmatrix)
tx=tvec[0] #- 5 #it's possible to change the translation value
ty=tvec[1] #- 5 #it's possible to change the translation value
tz=tvec[2]

print(tvec)

drawed=cv2.drawChessboardCorners(nimage,(7,7),corners,retval)
cv2.imshow("image",drawed)
cv2.waitKey(0)
cv2.destroyAllWindows()
