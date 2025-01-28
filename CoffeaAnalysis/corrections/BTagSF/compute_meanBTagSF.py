
import correctionlib
import numpy as np
import matplotlib.pyplot as plt
import os
from BTagSF.helpers import Corr_to_vec, save_hist_schemav2
from hist import Hist, axis

#parameters -----------------------------------------------------------------------------------------------------
period = '2017'
#----------------------------------------------------------------------------------------------------------------

channels = ['tee', 'tem', 'tte', 'ttm', 'tmm']
flavors = [0,4,5]

for flavor in flavors:
    print(f'For flavor {str(flavor)}')
    path = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/data/btagEff_json/{period}/')
    path_figure = os.path.join(os.getenv("ANALYSIS_PATH"), f'CoffeaAnalysis/corrections/BTagSF/figures/{period}/Jet_flavor_{str(flavor)}/diff_MeanVSchannel/')
    if not os.path.exists(path_figure):
        os.makedirs(path_figure)

    pt_bins = np.array( [20.,25., 30., 45.,75., 1000.] )
    eta_bins = np.array( [0,0.5,1.0,1.5,2.0,2.5] )
    # Create a meshgrid for the bin centers
    x_centers = np.array([22.5,27.5, 37.5, 60.,537.5]) #[22.5.,27.5., 37.5, 60.,537.5]
    y_centers = np.array(eta_bins[:-1]) + 0.25 #[0.25,0.75,1.25,1.75,2.25]

    # load sf:
    sf = {}
    sf_err = {}
    for channel in channels:
        file_name = f'btagEff_{channel}_{str(flavor)}.json'
        file_name_err = f'btagEff_{channel}_{str(flavor)}_err.json'
        f_path = os.path.join(path, file_name)
        f_path_err = os.path.join(path, file_name_err)
        ceval = correctionlib.CorrectionSet.from_file(f_path)
        ceval_err = correctionlib.CorrectionSet.from_file(f_path_err)
        BTag_eff = ceval["BTag_eff"]
        BTag_eff_err = ceval_err["BTag_eff_err"]
        BTag_eff_vec = Corr_to_vec(BTag_eff, x_centers, y_centers)
        BTag_eff_vec_err = Corr_to_vec(BTag_eff_err, x_centers, y_centers)
        sf[channel] = BTag_eff_vec 
        sf_err[channel] = BTag_eff_vec_err 

    # compute mean sf:
    mean_sf = sf['tee']
    mean_sf_err = sf_err['tee']*sf_err['tee']
    for channel in ['tem', 'tte', 'ttm','tmm']: 
        mean_sf = mean_sf + sf[channel]
        mean_sf_err = mean_sf_err + sf_err[channel]*sf_err[channel]
    mean_sf =  mean_sf/len(channels)
    mean_sf_err = np.sqrt(mean_sf_err)/len(channels)

    for channel in channels:
        print(f'--- plot channel {channel}')
        # Calculate the difference between the two corrections
        diff_corr = mean_sf - sf[channel]

        # Plot the difference as a 2D histogram
        plt.figure()
        plt.pcolormesh(pt_bins, eta_bins, abs(diff_corr), cmap='viridis', shading='auto')
        #plt.imshow(abs(diff_corr), origin='lower', extent=[pt_bins[0], pt_bins[-1], eta_bins[0], eta_bins[-1]], cmap='coolwarm')
        plt.colorbar()
        plt.xscale('log')
        plt.xlabel('pt[GeV]')
        plt.ylabel('abs(eta)')
        plt.title(f'{channel}')
        plt.savefig(os.path.join(path_figure,f'Diff_bTagEffMean_{channel}.pdf'))
        plt.close()

    #save mean_sf and mean_sf_err in a json file: f'btagEff_{str(flavor)}.json' as corrlib schéma 
    # Create histogram
    BTagEff_hist = Hist(axis.Variable(pt_bins, name='pt'), 
                        axis.Variable(eta_bins, name='abseta'))
    # Create histogram err
    BTagEff_hist_err = Hist(axis.Variable(pt_bins, name='pt'), 
                        axis.Variable(eta_bins, name='abseta'))
    
    # Fill histogram with values
    BTagEff_hist.fill(pt=np.repeat(pt_bins[:-1], len(eta_bins)-1),
        abseta=np.tile(eta_bins[:-1], len(pt_bins)-1),
        weight=mean_sf.flatten())
    # Fill histogram err with values
    BTagEff_hist_err.fill(pt=np.repeat(pt_bins[:-1], len(eta_bins)-1),
        abseta=np.tile(eta_bins[:-1], len(pt_bins)-1),
        weight=mean_sf_err.flatten())
    
    # without a name, the resulting object will fail validation
    BTagEff_hist.name = "BTagEff"
    BTagEff_hist.label = "BTagEff"
    BTagEff_hist_err.name = "BTagEff_err"
    BTagEff_hist_err.label = "BTagEff_err"

    #convert
    BTagEff_corr = correctionlib.convert.from_histogram(BTagEff_hist)
    BTagEff_corr_err = correctionlib.convert.from_histogram(BTagEff_hist_err)

    # set overflow bins behavior (default is to raise an error when out of bounds)
    BTagEff_corr.data.flow = "clamp"
    BTagEff_corr_err.data.flow = "clamp"

    # Save CorrectionSet to a JSON file
    output = os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/data/btagEff_json/{period}/btagEff_{str(flavor)}.json')
    output_err =  os.path.join(os.getenv("ANALYSIS_PATH"),f'CoffeaAnalysis/corrections/BTagSF/data/btagEff_json/{period}/btagEff_{str(flavor)}_err.json')
    save_hist_schemav2(BTagEff_corr, "Jet BTag efficiency (Loose WP)", output)
    save_hist_schemav2(BTagEff_corr_err, "Jet BTag efficiency error (Loose WP)", output_err)
