import numpy as np
from scipy import misc
import json
import time
import os
import NMF_ADMM_LR as odin


def loadFrames(path_in, frame_prefix, frame_start, frame_end, frame_format):

    # load one frame
    img0 = misc.imread('{0}/{1}{2}{3}'.format(path_in, frame_prefix, frame_start, frame_format)).astype('float64')

    # get dimensions
    r, c, d = img0.shape

    # define matrix
    A = np.zeros((r * c, frame_end - frame_start + 1))

    # load all frames
    for i in range(frame_start, frame_end+1):
        img = misc.imread('{0}/{1}{2}{3}'.format(path_in, frame_prefix, i, frame_format)).astype('float64')
        A[:, i-frame_start] = np.mean(img, 2).flatten()

    # Make A_ij \in [0,1]
    max_A = np.max(np.abs(A))
    A = (1.0 / max_A) * A

    return A, r, c


def generateFrames(A, r, c, path_out, file_name, frame_start, frame_end, frame_format):

    max_A = np.max(np.abs(A))
    A = (255.0 / max_A) * A

    for i in range(frame_start, frame_end + 1):
        name = '{0}/{1}{2}{3}'.format(path_out, file_name, i, frame_format)
        misc.imsave(name, A[:, i-frame_start].reshape((r, c)).astype('int'))


def mixFrames(M, L, S, r, c, path_out, file_name, frame_start, frame_end, frame_format):

    max_M = np.max(M)
    M = (255.0 / max_M) * M

    max_L = np.max(L)
    L = (255.0 / max_L) * L

    max_S = np.max(np.abs(S))
    S = (255.0 / max_S) * S

    for i in range(frame_start, frame_end + 1):

        X1 = M[:, i-frame_start].reshape((r, c)).astype('int')
        X2 = L[:, i-frame_start].reshape((r, c)).astype('int')
        X3 = S[:, i-frame_start].reshape((r, c)).astype('int')
        X = np.concatenate((X1, X2, X3), axis=1)
        misc.imsave('{0}/{1}{2}{3}'.format(path_out, file_name, i, frame_format), X)


if __name__ == '__main__':

    # load configuration file
    config = json.loads(open('./video-config.json').read())

    # execute each experiment defined in the configuration file
    for test in config["Execute"]:

        # Load config arguments
        path_in = config["Experiments"][test]["files"]["path_in"]
        path_out = os.path.join(config["Experiments"][test]["files"]["path_out"], test)
        frame_prefix = config["Experiments"][test]["files"]["frame_prefix"]
        frame_start = config["Experiments"][test]["files"]["frame_range0"]
        frame_end = config["Experiments"][test]["files"]["frame_range1"]
        frame_format = config["Experiments"][test]["files"]["frame_format"]

        iters = config["Experiments"][test]["parameters"]["iters"]
        mu1 = config["Experiments"][test]["parameters"]["mu1"]
        mu2 = config["Experiments"][test]["parameters"]["mu2"]
        epsilon = config["Experiments"][test]["parameters"]["epsilon"]
        rho = config["Experiments"][test]["parameters"]["rho"]

        # load images and generate matrix A with each frame per column
        A, r, c = loadFrames(path_in, frame_prefix, frame_start, frame_end, frame_format)

        '''
        # execute algorithm M = L + S
        t0 = time.clock()
        W, d, H, S = odin.mls_WDH(A, mu1, mu2, epsilon, rho, iters)
        t1 = time.clock()
        '''
        # execute algorithm NMF-LR
        t0 = time.clock()
        W, d, H, S = odin.nmf_WDH_LS(A, mu1, mu2, rho, iters)
        t1 = time.clock()


        # create directory
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # save statistics for this experiment
        statistics = {
            "Experiment": config["Experiments"][test],
            "Results": {
                "time": t1 - t0,
                "rank": d.shape[0],
                "relError": np.linalg.norm(A - np.dot(W, np.dot(np.diag(d), H)) - S)**2 / np.linalg.norm(A)**2
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

        # save frames from L and S
        L = np.dot(W, np.dot(np.diag(d), H))
        generateFrames(L, r, c, path_out, 'L_', frame_start, frame_end, frame_format)

        generateFrames(S, r, c, path_out, 'S_', frame_start, frame_end, frame_format)

        mixFrames(A, L, S, r, c, path_out, 'X_', frame_start, frame_end, frame_format)
