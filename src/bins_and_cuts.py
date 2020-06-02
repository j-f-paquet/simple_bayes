#!/usr/bin/env python3
import numpy as np

pT_bins = np.array( [ [0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70] ] ) # 8 bins

#the observables which will be used for parameter estimation
obs_cent_list = {

'Au-Au-200': {
	'RAA_charged_hadrons' : pT_bins,
    },

#'Au-Au-200': {
#	#'dNch_deta' : STAR_cent_bins, #unavailable
#	#'dET_deta' : STAR_cent_bins, #unavailable
#	'dN_dy_pion'   : central_STAR_cent_bins,
#	'dN_dy_kaon'   : central_STAR_cent_bins,
#	#current calculations use STAR centrality bins
#	#NOTE that the model calculations need to be re-averaged using the PHENIX cent bins if we want to include proton 
#	'dN_dy_proton' : central_STAR_cent_bins,
#	#'dN_dy_Lambda' : np.array([[0,5],[5,10],[10,20],[20,40],[40,60]]), #unavailable
#	#'dN_dy_Omega'  : np.array([[0,10],[10,20],[20,40],[40,60]]), #unavailable
#	#'dN_dy_Xi'     : np.array([[0,10],[10,20],[20,40],[40,60]]), #unavailable
#	'mean_pT_pion'   : central_STAR_cent_bins,
#	'mean_pT_kaon'   : central_STAR_cent_bins,
#	'mean_pT_proton' : central_STAR_cent_bins,
#	#'pT_fluct' : STAR_cent_bins, #unavailable
#	'v22' : central_STAR_cent_bins,
#	'v32' : central_STAR_cent_bins,
#	#'v42' : STAR_cent_bins,
#    },


}

#these just define some 'reasonable' ranges for plotting purposes
obs_range_list = {
	'Au-Au-200': {
		'RAA_charged_hadrons': [0,1],
    },
}
