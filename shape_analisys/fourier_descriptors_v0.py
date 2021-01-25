import cv2
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

def calc_centroid_distance(contour):
    moments = cv2.moments(contour)
    c_row=moments["m01"]/moments["m00"]
    c_col=moments["m10"]/moments["m00"]
    print("{},{}".format(c_col,c_row))
    contour_array = np.array(contour)
    contour_array = np.resize(contour_array,(-1,2))
    d_col = contour_array[:,0] - c_col
    d_row = contour_array[:,1] - c_row
    c_funct = np.sqrt(np.square(d_col) + np.square(d_row))
    return c_funct

def calc_fourier_descriptor(c_function):
    fourier_desc = []
    N = len(c_function)
    for u in range(N):
        sum = 0
        for k in range(N):
            sum = sum + c_function[k]*np.exp((-1j*2*np.pi*u*k)/N)
        fourier_desc.append(sum/N)
    return fourier_desc

def calc_fourier_descriptor2(c_function):
    fourier_desc = fft(c_function)
    return fourier_desc

def calc_normalized_fourier(contour):
    c_funct = calc_centroid_distance(contour)
    fourier_desc = calc_fourier_descriptor2(c_funct)
    normalized_descriptors = fourier_desc/fourier_desc[1]
    normalized_descriptors = np.abs(normalized_descriptors)
    return normalized_descriptors

def compare_fourier_descriptors(fourier1,fourier2,N=10):
    Fourier1 = fourier1[1:N+1]
    Fourier2 = fourier2[1:N+1]
    error = Fourier1 - Fourier2
    error_sq = np.square(error)
    rms_error = np.sqrt(np.sum(error_sq)/N)
    return rms_error

img = cv2.imread("one_leave2.png",1)
mask = cv2.imread("one_leave2.png",0)

contours,hie = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
print(len(contours))
descriptors = calc_normalized_fourier(contours[0])
#descriptors1 = calc_normalized_fourier(contours[1])
#descriptors2 = calc_normalized_fourier(contours[2])
#descriptors3 = calc_normalized_fourier(contours[3])
np.save("4leaves_shape.npy",descriptors)


#plt.plot(descriptors)
#plt.plot(descriptors1)
#plt.plot(descriptors2)
#plt.plot(descriptors3)
#plt.show()


#new_img = cv2.drawContours(img,contours,-1,(0,0,255))

#cv2.imshow("drawed",new_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()