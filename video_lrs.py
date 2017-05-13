import numpy as np
from scipy import misc
import NMF_ADMM_LR as odin


def loadFrames(frameNames, path):

    # number of frames
    f = len(frameNames)

    # load one image
    img0 = misc.imread('./{0}/{1}'.format(path,frameNames[0])).astype('float64')

    # get dimension
    w, h, d = img0.shape

    # define matrix
    A = np.zeros((w * h, f))

    for i in range(f):
        img = misc.imread('./{0}/{1}'.format(path,frameNames[i])).astype('float64')
        A[:, i] = np.mean(img, 2).flatten()

    return A, w, h


def generateFrames(A, w, h, fileName):

    for i in range(A.shape[1]):
        misc.imsave('./demo_out/{0}{1}.png'.format(fileName, i + 1), A[:, i].reshape((w, h)).astype('int'))


if __name__ == '__main__':

    # Define frames from video
    path = './demo'
    frameNames = ['demo_1.png', 'demo_2.png', 'demo_3.png', 'demo_4.png', 'demo_5.png', 'demo_6.png', 'demo_7.png',
                  'demo_8.png', 'demo_9.png', 'demo_10.png', 'demo_11.png', 'demo_12.png', 'demo_13.png', 'demo_14.png',
                  'demo_15.png', 'demo_16.png', 'demo_17.png', 'demo_18.png', 'demo_19.png', 'demo_20.png',
                  'demo_21.png', 'demo_22.png', 'demo_23.png', 'demo_24.png', 'demo_25.png', 'demo_26.png',
                  'demo_27.png', 'demo_28.png', 'demo_29.png', 'demo_30.png', 'demo_31.png', 'demo_32.png',
                  'demo_33.png', 'demo_34.png', 'demo_35.png', 'demo_36.png', 'demo_37.png', 'demo_38.png',
                  'demo_39.png', 'demo_40.png', 'demo_41.png', 'demo_42.png', 'demo_43.png', 'demo_44.png',
                  'demo_45.png', 'demo_46.png', 'demo_47.png', 'demo_48.png', 'demo_49.png', 'demo_50.png',
                  'demo_51.png']

    # load images and generate matrix A with each frame per column
    A, w, h = loadFrames(frameNames,path)

    # Make A_ij \in [0,1]
    max_A = np.max(A)
    A_norm = (1.0 / max_A) * A

    # Define parameters
    iters = [50, 100, 200, 500, 1000]
    mu1 = [15., 10., 10., 5., 5., 5.]
    mu2 = [0.001, 0.001, 0.001, 0.001, 0.001]
    epsilon = [0.0001, 0.0001, 0.0001, 0.0001, 0.00000001]
    rho = 1.0

    # Execute algorithm M = L + S
    import time
    t0 = time.clock()
    W, d, H, S = odin.mls_WDH(A_norm, mu1, mu2, epsilon, rho, iters)
    t1 = time.clock()

    # Print some result
    print 'time: ', t1 - t0
    print('rank: ', d.shape)
    print('minFx: ', np.linalg.norm(A_norm - np.dot(W, np.dot(np.diag(d), H)) - S))

    # Save frames from L and S
    generateFrames(max_A*(np.dot(W, np.dot(np.diag(d), H))), w, h, 'L')
    generateFrames(max_A * np.abs(S), w, h, 'S')