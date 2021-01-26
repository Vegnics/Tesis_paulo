import cv2
import numpy as np
from numpy.fft import fft2,fftshift,ifft2,ifftshift
import matplotlib.pyplot as plt


def click_on_rgb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        H = I_filtered_HSV[y,x,0]
        S = I_filtered_HSV[y,x,1]
        V = I_filtered_HSV[y,x,2]
        print("H={}, S={}, V={}".format(H,S,V))

def highpass_gaussian_kernel(size0,size1,sigma):
    kernel = np.zeros((size0,size1))
    for i in range(size0):
        for j in range(size1):
            kernel[i,j] = 1 - np.exp(-((i-int(size0/2))**2 + (j-int(size1/2))**2)/(2*sigma**2))
    return kernel

def highpass_butterworth_kernel(size0,size1,sl,sh,rc,n):
    kernel = np.zeros((size0,size1))
    for i in range(size0):
        for j in range(size1):
            kernel[i,j] = sh + (sl-sh)/(1+2.415*(((((i-int(size0/2))**2 + (j-int(size1/2))**2)**0.5)/rc))**(2*n))
    return kernel

def lowpass_gaussian_kernel(size0,size1,sigma):
    kernel = np.zeros((size0,size1))
    for i in range(size0):
        for j in range(size1):
            kernel[i,j] = np.exp(-((i-int(size0/2))**2 + (j-int(size1/2))**2)/(2*sigma**2))
    return kernel

def remap(src,min,max):
    #output_img=(255/(max-min))*(src-min)
    #output_img=output_img
    #output_img = np.clip(output_img,0,255)
    #output_img = output_img.astype(np.uint8)
    ##ADDED
    output_img = np.clip(src, 0, 255)
    output_img = output_img.astype(np.uint8)
    output_img = cv2.equalizeHist(output_img)
    ##ADDED##
    #plot_histogram(output_img)
    return output_img

def homomorph_filter_N1(src,kernel):
    src = src.astype(np.float32)
    Ln_I = np.log(src + 1)
    I_fft = fft2(Ln_I)
    I_fft = fftshift(I_fft)
    #kernel = highpass_gaussian_kernel(I_fft.shape[0], I_fft.shape[1], sigma)
    I_filt_fft = I_fft * kernel
    I_filt_fft_uns = ifftshift(I_filt_fft)
    I_filtered = np.real(ifft2(I_filt_fft_uns))
    I_filtered = np.exp(I_filtered) - 1
    return I_filtered,np.min(I_filtered),np.max(I_filtered)

def homomorph_filter_N3(src,kernel):
    B, G, R = cv2.split(src)
    nB,minB,maxB = homomorph_filter_N1(B, kernel)
    nG,minG,maxG = homomorph_filter_N1(G, kernel)
    nR,minR,maxR = homomorph_filter_N1(R, kernel)
    #plot_histogram(nR)
    max=np.max([maxB,maxG,maxR])
    min=np.min([minB,minG,minR])
    #nB=remap(nB,minB,maxB)
    #nG = remap(nG, minG, maxB)
    nB=  remap(nB, min, max)
    nG = remap(nG, min, max)
    nR = remap(nR, min, max)
    return cv2.merge((nB,nG,nR))

def plot_kernel(kernel):
    U=np.arange(0,kernel.shape[0])
    V=np.arange(0,kernel.shape[1])
    V,U = np.meshgrid(V,U)
    Z=kernel
    #axes = fig.gca(projection='3d')
    axes=plt.axes(projection='3d')
    axes.set_xlabel("s")
    axes.set_ylabel("t")
    axes.set_zlabel("H(s,t)")
    axes.plot_surface(V,U,Z,edgecolor='yellow')
    plt.show()

def plot_histogram(src):
    reshaped = np.resize(src,(-1,1))
    n, bins, patches = plt.hist(x=reshaped , bins='auto', color='blue',alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    #plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def nothing(x):
    pass

def remove_noise(mask,thresh):
    new_mask=np.zeros(mask.shape)
    num_labels, labeled = cv2.connectedComponents(mask)
    for label in range(num_labels):
        num_pix = np.where(labeled==label)
        if num_pix[0].size>thresh:
            aux_mask = np.where((labeled==label)&(labeled>0),255,0)
            new_mask = new_mask + aux_mask
            new_mask = new_mask.astype(np.uint8)
    return new_mask

def hole_filling(mask,thresh):
    mask = mask.copy()
    new_mask = np.zeros(mask.shape,dtype=np.uint8)
    mask_inv = cv2.bitwise_not(mask)
    mask_out = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    num_labels, labeled = cv2.connectedComponents(mask_inv)
    for label in range(num_labels):
        coords = zip(*np.where(labeled==label))
        coords = list(coords)
        print(len(coords))
        if len(coords)<thresh:
            #aux_mask = np.where((labeled==label),255,0)
            #new_mask = new_mask + aux_mask
            #new_mask = new_mask.astype(np.uint8)
            mask_out_aux = np.zeros(mask_out.shape, dtype=np.uint8)
            cv2.floodFill(mask_inv, mask_out_aux, coords[0][::-1], 255)
            new_mask = new_mask | mask_out_aux[1:-1,1:-1]
    new_mask = new_mask | mask
    return new_mask
#cv2.setMouseCallback("filtered", click_on_rgb)

#folder = "/home/amaranth/Desktop/TESIS/Tesis_paulo/database/class_c/"
#image_name="c_1.jpeg"
img_rgb = cv2.imread("img1.jpg",1)
#img_rgb = cv2.imread(folder+image_name,1)
kernel = highpass_butterworth_kernel(img_rgb.shape[0],img_rgb.shape[1],0.8,1.1,55,1)
#kernel = highpass_gaussian_kernel(img_rgb.shape[0],img_rgb.shape[1],40)
I_filtered = homomorph_filter_N3(img_rgb,kernel)
I_filtered_HSV = cv2.cvtColor(I_filtered,cv2.COLOR_BGR2HSV)
#plot_kernel(kernel)

H=[20,90]
S=[25,255]
V=[40,255]

img_copy = img_rgb.copy()

kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

mask = cv2.inRange(I_filtered_HSV,np.array([H[0],S[0],V[0]]),np.array([H[1],S[1],V[1]]))
img_segmented= cv2.bitwise_and(img_rgb,img_rgb,mask=mask)


mask = remove_noise(mask,300)
kernel_morph2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,kernel_morph2,iterations=5)
mask = hole_filling(mask,1000)
mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE,kernel_morph2,iterations=3)

img_segmented2 = cv2.bitwise_and(img_copy,img_copy,mask=mask)


cv2.imshow("segmented_with_noise",img_segmented)
cv2.imshow("segmented",img_segmented2)
cv2.imshow("original",img_rgb)
#cv2.imshow("segmented",img_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()



