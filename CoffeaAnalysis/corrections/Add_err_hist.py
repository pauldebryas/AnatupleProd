# For corrections: add correction error in a separate TTree so that we can use correctionlib library (no need to import ROOT: make the code faster)
import ROOT
import os

#parameters
# Path to the ROOT file
year = '2018'
Root_file_path = 'electron/sf_el_2018_HLTEle32.root' 
new_root_file_path = 'electron/sf_el_2018_HLTEle32_witherr.root'


correction_folder_path = f'{os.getenv("ANALYSIS_PATH")}/CoffeaAnalysis/corrections/{year}/'
root_file = ROOT.TFile.Open(correction_folder_path+Root_file_path)

# Access the histogram from the ROOT file
histogram = root_file.Get("SF2D")

# Create a new histogram with the same properties as the original one
new_histogram = ROOT.TH2F(histogram.GetName() + "_err",
                          histogram.GetTitle(),
                          histogram.GetNbinsX(),
                          histogram.GetXaxis().GetXmin(),
                          histogram.GetXaxis().GetXmax(),
                          histogram.GetNbinsY(),
                          histogram.GetYaxis().GetXmin(),
                          histogram.GetYaxis().GetXmax())

# Create a new ROOT file for the copy
new_file = ROOT.TFile.Open(correction_folder_path+new_root_file_path, "RECREATE")

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
        new_histogram.SetBinContent(binx, biny, bin_error)
        print('New Bin value: ', new_histogram.GetBinContent(binx, biny))
        print('')

histogram.Write()
new_histogram.Write()

# Remember to close the ROOT file when you are done
new_file.Close()
root_file.Close()