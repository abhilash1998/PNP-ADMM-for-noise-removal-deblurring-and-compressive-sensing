#!/usr/bin/env python
# coding: utf-8

import numpy as np
import bm3d
import matplotlib.pyplot as plt


from PIL import Image

import  os
cwd=os.getcwd()
icwd=cwd+"\\house_small.png"

im=Image.open(icwd)
ref=im.copy()
im=np.array(im)

#im.resize((64,64))
im.shape
#im=im.resize((4096,1))
#im.shape



im=im.flatten()


im=im.reshape(im.shape[0],1)
im.shape
im_vector=im









phi=np.random.normal(0,np.sqrt(1/4096),(4096,128*128))
phi2=np.random.normal(0,np.sqrt(1/8192),(8192,128*128))
#phi.shape



#y=np.dot(phi,im)
y=phi@im
y2=phi2@im



#phi.shape



X=np.zeros(im_vector.shape)

U=np.zeros(im_vector.shape)
V=np.zeros(im_vector.shape)

X_tilda=V-U




#X.shape



rho=1

std_dev=30

inv=np.linalg.inv((phi.T@phi)+rho*np.identity(phi.shape[1]))
print(inv.shape)

for i in range(100):
    X_tilda=V-U

    X=inv@((phi.T@y)+rho*X_tilda)
    X_U=(X+U)
    X_U.resize((int(np.sqrt(X.shape[0])),int(np.sqrt(X.shape[0]))))
    X_U.resize
    
    V= bm3d.bm3d(X_U, std_dev)
    
    V=V.flatten()
    V.resize(V.shape[0],1)
    U=U+(X-V)
    





X2=np.zeros(im_vector.shape)

U=np.zeros(im_vector.shape)
V=np.zeros(im_vector.shape)
#X=np.zeros(im.shape)
#X=s
# X=np.zeros(s.shape)
# U=np.zeros(s.shape)
# V=np.zeros(s.shape)
X_tilda=V-U
#std_dev=0.01
rho=1
#std_dev=np.sqrt(lambd/rho)

std_dev=30

inv=np.linalg.inv((phi2.T@phi2)+rho*np.identity(phi2.shape[1]))

for i in range(100):
    X_tilda=V-U
    

    X2=inv@((phi2.T@y2)+rho*X_tilda)
    X_U=(X2+U)
    X_U.resize((int(np.sqrt(X.shape[0])),int(np.sqrt(X.shape[0]))))
    X_U.resize
    #print(X_U.shape)
    V= bm3d.bm3d(X_U, std_dev)
    #X=X.flatten()
    V=V.flatten()
    V.resize(V.shape[0],1)
    U=U+(X2-V)
    







#X.resize(int(np.sqrt(X.shape[0])),int(np.sqrt(X.shape[0])))
X.resize(128,128)
X2.resize(128,128)



plt.subplot(1, 2, 1)
plt.imshow(X, 'gray')
plt.subplot(1, 2, 2)
plt.imshow(X2, 'gray')
plt.show()


import math
def PSNR(original, x):
    meanse = np.mean((original - x) ** 2)
    if(meanse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / np.sqrt(meanse))
    return psnr







print("PSNR for 1st condition",PSNR(ref,X))


print("PSNR for 2nd condition",PSNR(ref,X2))

