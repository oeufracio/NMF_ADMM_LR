import numpy as np
import scipy as sp
from scipy import misc


def loadFrames( frameNames ):

    # number of frames
    f = len(frameNames)
        
    # load one image
    img0 = misc.imread('./{0}'.format(frameNames[0])).astype('float64')
    
    # get dimmension
    w, h, d = img0.shape

    # define matrix 
    A = np.zeros((w*h,f))

    for i in range(f):
        img = misc.imread('./{0}'.format(frameNames[i])).astype('float64')
        A[:,i] = np.mean(img,2).flatten()
    
    return A, w, h
    

def generateFrames( A, w, h):

    for i in range(A.shape[1]):
        misc.imsave('L{0}.png'.format(i+1), A[:,i].reshape((w,h)).astype('int'))
    

if __name__ == '__main__':

    frameNames = ['demo_1.png', 'demo_2.png', 'demo_3.png', 'demo_4.png', 'demo_5.png', 'demo_6.png', 'demo_7.png', 'demo_8.png', 'demo_9.png', 'demo_10.png', 'demo_11.png', 'demo_12.png', 'demo_13.png', 'demo_14.png', 'demo_15.png', 'demo_16.png', 'demo_17.png', 'demo_18.png', 'demo_19.png', 'demo_20.png', 'demo_21.png', 'demo_22.png', 'demo_23.png', 'demo_24.png', 'demo_25.png', 'demo_26.png', 'demo_27.png', 'demo_28.png', 'demo_29.png', 'demo_30.png', 'demo_31.png', 'demo_32.png', 'demo_33.png', 'demo_34.png', 'demo_35.png', 'demo_36.png', 'demo_37.png', 'demo_38.png', 'demo_39.png', 'demo_40.png', 'demo_41.png', 'demo_42.png', 'demo_43.png', 'demo_44.png', 'demo_45.png', 'demo_46.png', 'demo_47.png', 'demo_48.png', 'demo_49.png', 'demo_50.png', 'demo_51.png']

    A, w, h = loadFrames(frameNames)

    generateFrames(A, w, h)






