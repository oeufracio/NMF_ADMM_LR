import numpy as np
from scipy import misc
import json
import time
import os
import NMF_ADMM_LR as odin


def loadFrames(path_in, faces, frame_start, frame_end, frame_format):

    # number of sets
    n = len(faces)

    # load one frame
    img0 = misc.imread('{0}/{1}/{2}{3}'.format(path_in, faces[0], frame_start, frame_format)).astype('float64')

    # get dimensions
    r, c = img0.shape

    # define matrix
    A = np.zeros( (r * c, n*(frame_end - frame_start + 1)) )
    
    count = 0

    #iterate each face directory
    for j in range(n):

        cur_directory = faces[j]

        # load images from current directory
        for i in range(frame_start, frame_end+1):
            img = misc.imread('{0}/{1}/{2}{3}'.format(path_in, cur_directory, i, frame_format)).astype('float64')
            A[:, count] = img.flatten()
            count += 1

    # Make A_ij \in [0,1]
    max_A = np.max(np.abs(A))
    A = (1.0 / max_A) * A

    return A, r, c


def generateFrames(A, r, c, path_out, file_name, frame_start, frame_end, frame_format):

    n = A.shape[1]

    max_A = np.max(np.abs(A))
    A = (255.0 / max_A) * A


    for i in range(n):
        name = '{0}/{1}{2}{3}'.format(path_out, file_name, i, frame_format)
        misc.imsave(name, A[:, i].reshape((r, c)).astype('int'))

def generateBasis(W, r, c, path_out, file_name, frame_format):

    k = W.shape[1]

    for i in range(k):
        max_W = np.max(W[:,i])
        W[:,i] = (255.0 / max_W) * W[:,i]
    
    
    X = W[:,0].reshape((r,c)).astype('int')

    for i in range(1,k):
        X = np.concatenate(( X,W[:,i].reshape((r,c)).astype('int')), axis=1)

    misc.imsave('{0}/{1}{2}'.format(path_out, file_name, frame_format), X)



def mixFrames(M, WH,  L, S, r, c, path_out, faces, frame_start, frame_end, frame_format):

    n = M.shape[1]

    max_M = np.max(M)
    M = (255.0 / max_M) * M

    max_L = np.max(L)
    L = (255.0 / max_L) * L

    max_S = np.max(np.abs(S))
    S = (255.0 / max_S) * S

    max_WH = np.max(WH)
    WH = (255.0 / max_WH) * WH


    for j in range( len(faces) ):

    # create directory
        if not os.path.exists( os.path.join(path_out,faces[j]) ):
            os.makedirs( os.path.join(path_out,faces[j]) )

        for i in range(frame_end-frame_start+1):

            X1 = M[:, j*(frame_end-frame_start+1)+i].reshape((r, c)).astype('int')
            X2 = WH[:,j*(frame_end-frame_start+1)+i].reshape((r, c)).astype('int')
            X3 = L[:, j*(frame_end-frame_start+1)+i].reshape((r, c)).astype('int')
            X4 = S[:, j*(frame_end-frame_start+1)+i].reshape((r, c)).astype('int')
            X5 = X3 + X4
            X = np.concatenate((X1, X2, X5, X3, X4), axis=1)
            misc.imsave('{0}/{1}{2}'.format(os.path.join(path_out,faces[j]), frame_start+i, frame_format), X)



if __name__ == '__main__':

    # load configuration file
    config = json.loads(open('./orl-config.json').read())

    # execute each experiment defined in the configuration file
    for test in config["Execute"]:

        # Load config arguments
        path_in = config["Experiments"][test]["files"]["path_in"]
        path_out = os.path.join(config["Experiments"][test]["files"]["path_out"], test)
        faces = config["Experiments"][test]["files"]["faces"]
        frame_start = config["Experiments"][test]["files"]["frame_range0"]
        frame_end = config["Experiments"][test]["files"]["frame_range1"]
        frame_format = config["Experiments"][test]["files"]["frame_format"]

        iters = config["Experiments"][test]["parameters"]["iters"]
        mu1 = config["Experiments"][test]["parameters"]["mu1"]
        mu2 = config["Experiments"][test]["parameters"]["mu2"]
        epsilon = config["Experiments"][test]["parameters"]["epsilon"]
        rho = config["Experiments"][test]["parameters"]["rho"]

        
        # load images and generate matrix A with each frame per column
        A, r, c = loadFrames(path_in, faces, frame_start, frame_end, frame_format)

        # ------------------------------------------------------- #
        # execute algorithm NMF-LR
        t0 = time.clock()
        W1, d1, H1, S1 = odin.nmf_WDH_LS(A.copy(), mu1, mu2, rho, iters)
        t1 = time.clock()

        #e execute algorithm NMF-ADMM
        W2, H2 = odin.nmf_WH(A.copy(), W1.shape[1], rho, int(3000))
        # ------------------------------------------------------- #


        # create directory
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # save statistics for this experiment
        statistics = {
            "Experiment": config["Experiments"][test],
            "Results": {
                "time": t1 - t0,
                "rank": d1.shape[0],
                "relError": np.linalg.norm(A - np.dot(W1, np.dot(np.diag(d1), H1)) - S1)**2 / np.linalg.norm(A)**2
            }
        }

        with open('{0}/{1}_results.json'.format(path_out,test), 'w') as outfile:
            json.dump(statistics, outfile, indent=4, sort_keys=True)


        # print some result
        print test
        print 'time: ', statistics["Results"]["time"]
        print 'rank: ', statistics["Results"]["rank"]
        print 'relError: ', statistics["Results"]["relError"]
        print
        print np.linalg.norm(A - np.dot(W2, H2))**2 / np.linalg.norm(A)**2

        
        # save W1
        generateBasis(W1, r, c, path_out,  'W1_', frame_format)
    
        # save W2
        generateBasis(W2, r, c, path_out,  'W2_', frame_format)


        # save reconstructions
        mixFrames(A, np.dot(W2, H2), np.dot(W1, np.dot(np.diag(d1), H1)), S1, r, c, path_out, faces, frame_start, frame_end, frame_format)
        
        
