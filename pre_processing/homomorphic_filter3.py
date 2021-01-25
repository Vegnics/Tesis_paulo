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

def remove_noise(mask,thresh):
    noise_pts=[]
    new_mask=mask.copy()
    num_labels, labeled = cv2.connectedComponents(mask)
    for label in range(num_labels):
        binarized_coords = zip(*np.where(labeled==label))
        binarized_coords = list(binarized_coords)
        if len(binarized_coords)<thresh:
            noise_pts = noise_pts + binarized_coords
    for pnt in noise_pts:
        new_mask[pnt]=0
    return new_mask

def nothing(x):
    pass

cv2.namedWindow("filtered")

cv2.createTrackbar('H_min','filtered',0,180,nothing)
cv2.createTrackbar('S_min','filtered',0,255,nothing)
cv2.createTrackbar('V_min','filtered',0,255,nothing)
cv2.createTrackbar('H_max','filtered',0,180,nothing)
cv2.createTrackbar('S_max','filtered',0,255,nothing)
cv2.createTrackbar('V_max','filtered',0,255,nothing)


#cv2.setMouseCallback("filtered", click_on_rgb)

folder = "/home/amaranth/Desktop/TESIS/Tesis_paulo/database/class_b/"
image_name="b_1.jpeg"
img_rgb1 = cv2.imread("/home/amaranth/Desktop/TESIS/Tesis_paulo/pre_processing/img1.jpg",1)
#img_rgb1 = cv2.imread(folder+image_name,1)
kernel = highpass_butterworth_kernel(img_rgb1.shape[0],img_rgb1.shape[1],0.8,1.1,55,1)
#kernel = highpass_gaussian_kernel(img_rgb.shape[0],img_rgb.shape[1],40)
I_filtered = homomorph_filter_N3(img_rgb1,kernel)
I_filtered_HSV = cv2.cvtColor(I_filtered,cv2.COLOR_BGR2HSV)

img_rgb2 = cv2.imread("/home/amaranth/Desktop/TESIS/Tesis_paulo/pre_processing/img2.jpg",1)
kernel = highpass_butterworth_kernel(img_rgb2.shape[0],img_rgb2.shape[1],0.8,1.1,55,1)
#kernel = highpass_gaussian_kernel(img_rgb.shape[0],img_rgb.shape[1],40)
I_filtered2 = homomorph_filter_N3(img_rgb2,kernel)
I_filtered_HSV2 = cv2.cvtColor(I_filtered2,cv2.COLOR_BGR2HSV)


#plot_kernel(kernel)

#H=[30,80]
#S=[50,255]
#V=[1,200]


#With histogram equalization
H=[27,100]
S=[50,255]
V=[32,255]
#img_contours = img_rgb.copy()

kernel_morph = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
stacked_rgb = np.hstack((img_rgb1,img_rgb2))
while(1):
    #H=[cv2.getTrackbarPos("H_min","filtered"),cv2.getTrackbarPos("H_max","filtered")]
    #S=[cv2.getTrackbarPos("V_min","filtered"),cv2.getTrackbarPos("V_max","filtered")]
    #V=[cv2.getTrackbarPos("S_min","filtered"),cv2.getTrackbarPos("S_max","filtered")]

    stacked_HSV = np.hstack((I_filtered_HSV,I_filtered_HSV2))
    mask = cv2.inRange(stacked_HSV ,np.array([H[0],S[0],V[0]]),np.array([H[1],S[1],V[1]]))
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel_morph,iterations=3)
    mask = remove_noise(mask,50)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel_morph, iterations=3)
    img_segmented = cv2.bitwise_and(stacked_rgb,stacked_rgb,mask=mask)
    #contours,hierar = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    #for cnt in contours:
    #    if len(cnt)>250:
    #        img_contours = cv2.drawContours(img_contours,cnt,-1,color=[0,0,255],thickness=3)

    cv2.imshow("filtered",img_segmented)
    cv2.imshow("original",stacked_rgb)
    #cv2.imshow("segmented",img_contours)
    k = cv2.waitKey(1)
    #cv2.destroyAllWindows()



