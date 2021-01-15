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

def remap(src,min,max):
    max=np.max(src)
    min=np.min(src)
    output_img=(255/(max-min))*(src-min)
    output_img=output_img
    output_img = np.clip(output_img,0,255)
    output_img = output_img.astype(np.uint8)
    return output_img

def homomorph_filter_N1(src,sigma):
    src = src.astype(np.float32)
    Ln_I = np.log(src + 1)
    I_fft = fft2(Ln_I)
    I_fft = fftshift(I_fft)
    kernel = highpass_gaussian_kernel(I_fft.shape[0], I_fft.shape[1], sigma)
    I_filt_fft = I_fft * kernel
    I_filt_fft_uns = ifftshift(I_filt_fft)
    I_filtered = np.real(ifft2(I_filt_fft_uns))
    I_filtered = np.exp(I_filtered)
    return I_filtered,np.min(I_filtered),np.max(I_filtered)

def homomorph_filter_N3(src,sigma):
    B, G, R = cv2.split(img_orig)
    nB,minB,maxB = homomorph_filter_N1(B, sigma)
    nG,minG,maxG = homomorph_filter_N1(G, sigma)
    nR,minR,maxR = homomorph_filter_N1(R, sigma)
    max=np.max([maxB,maxG,maxR])
    min=np.min([minB,minG,minR])
    nB=remap(nB,min,max)
    nG = remap(nG, min, max)
    nR = remap(nR, min, max)
    return cv2.merge((nB,nG,nR))



img_orig=cv2.imread("img_lighting.jpg",1)
img_orig=cv2.resize(img_orig,(-1,-1),fx=1/4,fy=1/4,interpolation=cv2.INTER_LINEAR)
#kernel = Create_gaussian_kernel(img_orig.shape[0],img_orig.shape[1],10)
#print(kernel)
#img=img_orig.astype(np.float32)+1
#log_img=np.log(img)
#log_filtered=cv2.filter2D(img,-1,kernel=kernel)
#log_sharpen=log_img-log_filtered
#exp_sharpen=np.exp(log_sharpen)
#filtered_img=remap(exp_sharpen)
"""
B,G,R = cv2.split(img_orig)
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
"""

I_filtered = homomorph_filter_N3(img_orig,1.1)
cv2.imshow("filtered",I_filtered)
cv2.imshow("original",img_orig)
cv2.waitKey(0)
cv2.destroyAllWindows()








