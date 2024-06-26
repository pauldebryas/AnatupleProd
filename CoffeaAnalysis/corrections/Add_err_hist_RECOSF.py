# For corrections: add correction error in a separate TTree so that we can use correctionlib library (no need to import ROOT: make the code faster)
import ROOT
import os
from array import array

#parameters
# Path to the ROOT file
year = '2018'
ptrange = 'ptBelow20'
# ------------------------

def create_histogram_like_original(original_hist, title):
    # Get bin edges along X-axis
    x_bins = [original_hist.GetXaxis().GetBinLowEdge(bin_idx) for bin_idx in range(1, original_hist.GetNbinsX() + 2)]
    # Get bin edges along Y-axis
    y_bins = [original_hist.GetYaxis().GetBinLowEdge(bin_idx) for bin_idx in range(1, original_hist.GetNbinsY() + 2)]
    
    x_bins_array = array('d', x_bins)
    y_bins_array = array('d', y_bins)

    # Create new histogram using bin edges
    new_histogram = ROOT.TH2F(title,
                              original_hist.GetTitle(),
                              len(x_bins) - 1, 
                              x_bins_array,
                              len(y_bins) - 1, 
                              y_bins_array)
    
    return new_histogram

file_name = {
        '2018': 'UL2018',
        '2017': 'UL2017',
        '2016': 'UL2016postVFP',
        '2016_HIPM': 'UL2016preVFP',
    }
Root_file_name = f'egammaEffi_{ptrange}.txt_EGM2D_{file_name[year]}.root' 
new_root_file_name = f'egammaEffi_{ptrange}-txt_EGM2D_{file_name[year]}_witherr.root'


correction_folder_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/{year}/electron/'
root_file = ROOT.TFile.Open(correction_folder_path+Root_file_name)

# Access the histogram from the ROOT file
histogram = root_file.Get("EGamma_SF2D")

# Create a new histogram with the same properties as the original one
new_histogram_err = create_histogram_like_original(histogram, "EGamma_SF2D_err")
new_histogram_val = create_histogram_like_original(histogram, 'EGamma_SF2D')

# Create a new ROOT file for the copy
new_file = ROOT.TFile.Open(correction_folder_path+new_root_file_name, "RECREATE")

# Get the number of bins in the X and Y directions
nbins_x = histogram.GetNbinsX()
nbins_y = histogram.GetNbinsY()

# Iterate over the bins and get the bin indices
for binx in range(1, nbins_x + 1):  # Loop over X bins
    for biny in range(1, nbins_y + 1):  # Loop over Y bins
        print(f'Bin ({binx},{biny})')
        # Get the bin value and error for a specific bin
        bin_error = histogram.GetBinError(binx, biny)
        bin_value = histogram.GetBinContent(binx, biny)
        # Use the bin valeu/error obtained from the histogram
        print("value:", bin_value)
        print("error:", bin_error)
        #if bin_value == 252451.5625:
        #    bin_value = 1.021
        #    bin_error = 0.1594
        #252452 --> 1.021
        #24455 --> 0.1594

        new_histogram_err.SetBinContent(binx, biny, bin_error)
        new_histogram_val.SetBinContent(binx, biny, bin_value)
        print('New Bin val: ', new_histogram_val.GetBinContent(binx, biny))
        print('New Bin err: ', new_histogram_err.GetBinContent(binx, biny))
        print('')

new_histogram_val.Write()
new_histogram_err.Write()

# Remember to close the ROOT file when you are done
new_file.Close()
root_file.Close()