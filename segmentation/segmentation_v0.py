import cv2
import numpy as np

folder = "/home/amaranth/Desktop/TESIS/Tesis_paulo/database/class_a/"
image_name="a_1.jpeg"
img_rgb = cv2.imread(folder+image_name,1)
img_hsv = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2HSV)

H=[25,75]
S=[35,255]
V=[0,255]

mask = cv2.inRange(img_hsv,np.array([H[0],S[0],V[0]]),np.array([H[1],S[1],V[1]]))
img_segmented = cv2.bitwise_and(img_rgb,img_rgb,mask=mask)
cv2.imshow("results",img_segmented)
cv2.imshow("original",img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()