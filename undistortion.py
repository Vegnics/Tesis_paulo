import cv2
import numpy as np

image_folder="/home/amaranth/Desktop/TESIS/Tesis_paulo/calibration_images/new_calibration_images/"
image=cv2.imread(image_folder+"extrinsic_calibration2.jpg")

intrinsics=np.load("intrinsics.npy")
distvec=np.load("distcoeffs.npy")

nintrinsics,roi=cv2.getOptimalNewCameraMatrix(intrinsics,distvec,(640,480),1,(640,480))
mapx,mapy= cv2.initUndistortRectifyMap(intrinsics,distvec,None,nintrinsics,(640,480),cv2.CV_32FC1)
np.save("mapx.npy",mapx)
np.save("mapy.npy",mapy)
np.save("nintrinsics.npy",nintrinsics)

nimage = cv2.remap(image,mapx,mapy,interpolation=cv2.INTER_LINEAR)
cv2.imshow("image",nimage)
cv2.waitKey(0)
cv2.destroyAllWindows()
