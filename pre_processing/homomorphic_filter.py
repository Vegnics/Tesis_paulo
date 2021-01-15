import cv2
import numpy as np
from numpy.fft import fft2,fftshift,ifft2,ifftshift

def highpass_gaussian_kernel(size0,size1,sigma):
    kernel = np.zeros((size0,size1))
    for i in range(size0):
        for j in range(size1):
            kernel[i,j] = 1 - np.exp(-((i-int(size0/2))**2 + (j-int(size1/2))**2)/(2*sigma**2))
    return kernel

def lowpass_gaussian_kernel(size0,size1,sigma):
    kernel = np.zeros((size0,size1))
    for i in range(size0):
        for j in range(size1):
            kernel[i,j] = np.exp(-((i-int(size0/2))**2 + (j-int(size1/2))**2)/(2*sigma**2))
    return kernel

def remap(src):
    max=np.max(src)
    min=np.min(src)
    output_img=(256/(max-min))*(src-min)
    output_img=output_img-1
    output_img = np.clip(output_img,0,255)
    output_img = output_img.astype(np.uint8)
    return output_img



img_orig=cv2.imread("img1.jpg",0)
#kernel = Create_gaussian_kernel(img_orig.shape[0],img_orig.shape[1],10)
#print(kernel)
#img=img_orig.astype(np.float32)+1
#log_img=np.log(img)
#log_filtered=cv2.filter2D(img,-1,kernel=kernel)
#log_sharpen=log_img-log_filtered
#exp_sharpen=np.exp(log_sharpen)
#filtered_img=remap(exp_sharpen)

img=img_orig.astype(np.float32)
Ln_I = np.log(img+1)
I_fft=fft2(Ln_I)
I_fft=fftshift(I_fft)
kernel= highpass_gaussian_kernel(I_fft.shape[0],I_fft.shape[1],2)
I_filt_fft=I_fft*kernel
I_filt_fft_uns=ifftshift(I_filt_fft)
I_filtered=np.real(ifft2(I_filt_fft_uns))
I_filtered = np.exp(I_filtered)
I_filtered = remap(I_filtered)
print(I_filtered)

cv2.imshow("filtered",I_filtered)
cv2.imshow("original",img_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()








