#!/usr/bin/env python3

import numpy as np
#import h5py
import sys, os, glob
# Output data format
from configurations import *
from bins_and_cuts import pT_bins_model

def list2array(func):
        def func_wrapper(x, w):
                try:
                        x = np.array(x)
                        w = np.array(w)
                except:
                        raise ValueError("cannot interpret input as numpy array...")
                return func(x, w)
        return func_wrapper

def make_fake_events(design_pt,system):

    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))

    idf_arr = [0]
    for idf in idf_arr:
        print("----------------------")
        print("idf : " + str(idf) )

        # dNdeta
        tmp_obs='RAA_charged_hadrons'
        try :
            pT=pT_bins_model 
            entry[system][tmp_obs]['mean'][:] = np.full(len(pT), [0.5,1.0,2.0][design_pt])
            entry[system][tmp_obs]['err'][:] = entry[system][tmp_obs]['mean'][:]*0.05
            #print(entry)
        except KeyError :
            pass

    return entry

if __name__ == '__main__':

    system = system_strs[0]

    print("Computing observables for all design points")
    print("System = " + system)
    n_design_pts_main=3
    system_str='Au-Au-200'
    n_design_pts_validation=0
    for file_output, nset in zip(
              [SystemsInfo[system_str]["main_obs_file"], SystemsInfo[system_str]["validation_obs_file"]],
              [n_design_pts_main, n_design_pts_validation],
           ):
        print("\n")
        print("##########################")
        results = []
        for i in range(nset):
            print("design pt : " + str(i))
            results.append(make_fake_events(i,system)[0])
            print("\n")
        results = np.array(results)
        print("results.shape = " + str(results.shape))
        results.tofile(os.path.join(".",file_output))
