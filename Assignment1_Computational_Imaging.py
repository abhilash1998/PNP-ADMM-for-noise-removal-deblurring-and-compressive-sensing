#!/usr/bin/env python
# coding: utf-8


import numpy as np
import bm3d


from PIL import Image


im=Image.open(r"C:\Users\abhil\Downloads\house_small.png")
im=np.array(im)


from scipy.fft import fft2, ifft2
fft=fft2
ifft=ifft2


def calculate_gaussian(length,sig):
    #length=3
    #sig=1
    k1= np.linspace(-(length - 1) / 2., (length - 1) / 2., length)
    x, y = np.meshgrid(k1, k1)

    ker = (np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sig)))/ np.sum((np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(sig))))

    return ker


kernel_1=calculate_gaussian(25,1)


kernel_1



kernel_3=calculate_gaussian(25,3)



im.shape



Fim=fft(im)



pad_size = (im.shape[0] - kernel_1.shape[0], im.shape[1] - kernel_1.shape[1])  # total amount of padding
kernel_1 = np.pad(kernel_1, (((pad_size[0]+1)//2, pad_size[0]//2), ((pad_size[1]+1)//2, pad_size[1]//2)), 'constant')
kernel_3 = np.pad(kernel_3, (((pad_size[0]+1)//2, pad_size[0]//2), ((pad_size[1]+1)//2, pad_size[1]//2)), 'constant')
#kernel = fftpack.ifftshift(kernel)
from scipy import fftpack
kernel_1=fftpack.ifftshift(kernel_1)
kernel_3=fftpack.ifftshift(kernel_3)


import matplotlib.pyplot as plt
plt.imshow(kernel_3,'gray')



from scipy import fftpack
#kernel_1 = fftpack.ifftshift(kernel_1)
Fkernel_1=fft(kernel_1)
Fkernel_3=fft(kernel_3)


Fim1=np.multiply(Fim,Fkernel_1)
Fim3=np.multiply(Fim,Fkernel_3)






im_3=(np.real(ifft(Fim3)))

im_1=(np.real(ifft(Fim1)))



noise=np.random.normal(0,1,im.shape)
im_1=im_1+noise
im_3=im_3+noise
#plt.imshow(im,'gray')

Image.fromarray(np.uint8(im_1))




#im_3=im_3/255



X=np.zeros(im_1.shape)

U=np.zeros(im_1.shape)
V=np.zeros(im_1.shape)
X_tilda=V-U
rho=1





std_dev=0.01



for i in range(100):
    X_tilda=V-U
    #X=np.real(ifft(np.divide((np.conj(fft(kernel_3))*fft(im_3)+ rho*fft(X_tilda)),((((fft(kernel_3))**2)+rho)))))
    X=np.real(ifft(np.divide((np.conj(fft(kernel_1))*fft(im_1)+ rho*fft(X_tilda)),((((fft(kernel_1)*fft(kernel_1)))+rho)))))
    V= bm3d.bm3d(X+U, std_dev)
    U=U+(X-V)
    #print(U)


plt.imshow(X,'gray')


#plt.imshow(im_1,'gray')


#plt.imshow(im,'gray')


X3=np.zeros(im_1.shape)

U=np.zeros(im_1.shape)
V=np.zeros(im_1.shape)
X_tilda=V-U
rho=0.1
std_dev=0.0001

















for i in range(100):
    X_tilda=V-U
    X3=np.real(ifft(np.divide((np.conj(fft(kernel_3))*fft(im_3)+ rho*fft(X_tilda)),((((fft(kernel_3)*fft(kernel_3)))+rho)))))
    #X3=np.real(ifft(np.divide((np.conj(fft(kernel_1))*fft(im_1)+ rho*fft(X_tilda)),((((fft(kernel_1)*fft(kernel_1)))+rho)))))
    V= bm3d.bm3d(X3+U, std_dev)
    U=U+(X3-V)
    #print(U)



plt.subplot(1, 2, 1)
plt.imshow(X, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(X3, 'gray')
plt.show()
#plt.imshow(X3,'gray')


#plt.imshow(im_3,'gray')




import math
def PSNR(original, x):
    meanse = np.mean((original - x) ** 2)
    if(meanse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / np.sqrt(meanse))
    return psnr



print("image2_blurr",PSNR(im,X3))






pritn("image1_blur",PSNR(im,X))


