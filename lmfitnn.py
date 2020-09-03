###### example syntax, to be deleted later 
######## linefit(wave,flux, 4861, 15, outputplot="Hbeta", model='VP',degree=4)
######### wave is the wavelength array, flux is flux array
######### 4861 is the Hbeta central wavelength, the line we want to fit
######### 15 is the line width, it is large because our resolution is quite poor
######### outplot is simply the outputplot name
######### model is the model chosen for the fit, currently choices are Gaussian (G), Double Gaussian (DG), both with a basleine polynomial fit (GP, or DGP), a
######## and same for Voigt (V), Double Voigt (DV) and with polynomial (VP, DVP). 
######## degree isthe degree of the polynomial baseline, 0 is default for nomrlized spectra
########## read in the wave and flux arrays from example file
########### you can also cut the input arrays to focus on just the line to see a better fit, i.e. mask the wavelnegth, flux arrays before running linefit

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, LinearModel, ConstantModel, PolynomialModel, ExponentialModel



    
def selectModel (model, wave, flux, ctr_wave, lw, degree=0):
    ########################### This function selects the model and determines intial fit parameters based on the central wavelength and linewidth
    ###########################
    #### Sanity checks
    if wave.min() > (ctr_wave-lw):
        raise Exception("Lower wavelength range does not cover the central line width")
    if wave.max() < (ctr_wave-lw):
        raise Exception("Upper wavelength range does not cover the central line width")
    ######################################################################################
    ###########################Polynomial baseline########################################
    ######################################################################################
    ######### Default is 0 assuming normalized spectra
    poly = PolynomialModel(degree=degree)
    pars = poly.guess(flux, x=wave)
    ######################################################################################
    ###########################Initial line guess#########################################
    ######################################################################################
    # Create mask around line
    lower_wl = ctr_wave - lw
    upper_wl = ctr_wave + lw
    maskline = np.ma.masked_outside(wave, lower_wl, upper_wl).mask
    line_wl = wave[~maskline]
    line_flux = flux[~maskline]
    # Run the inital guess
    guess = init_guess(line_wl,line_flux)
    ######################################################################################
    ###########################Gaussian Models############################################
    ######################################################################################
    Gaussian_1 = GaussianModel(prefix='G1_')
    ###### update the Gaussian parameters
    pars.update(Gaussian_1.make_params())
    pars['G1_center'].set(value=guess[2])
    pars['G1_sigma'].set(value=guess[1])
    pars['G1_amplitude'].set(value=guess[0])
    ###### update the Gaussian parameters
    Gaussian_2 = GaussianModel(prefix='G2_')
    pars.update(Gaussian_2.make_params())
    pars['G2_center'].set(value=guess[2])
    pars['G2_sigma'].set(value=guess[1])
    pars['G2_amplitude'].set(value=guess[0])
    

    ######################################################################################
    ###########################Voigt Models###############################################
    ######################################################################################
    Voigt_1 = VoigtModel(prefix='V1_')
    ######## update the Voigt parameters
    pars.update(Voigt_1.make_params())
    pars['V1_center'].set(value=guess[2])
    pars['V1_sigma'].set(value=guess[1])
    pars['V1_gamma'].set(value=guess[1])
    pars['V1_amplitude'].set(value=guess[0])
    ######## update the Voigt parameters
    Voigt_2 = VoigtModel(prefix='V2_')
    pars.update(Voigt_2.make_params())
    pars['V2_center'].set(value=guess[2])
    pars['V2_sigma'].set(value=guess[1])
    pars['V2_gamma'].set(value=guess[1])
    pars['V2_amplitude'].set(value=guess[0])

    ######################################################################################
    ###########################Model Dict#################################################
    ######################################################################################

    
    options ={
    'G': Gaussian_1,
    'GP':Gaussian_1 + poly,
    'DG': Gaussian_1 + Gaussian_2,
    'DGP': Gaussian_1 + Gaussian_2 + poly,
    'V': Voigt_1,
    'VP': Voigt_1 + poly,
    'DV': Voigt_1 + Voigt_2,
    'DVP': Voigt_1 + Voigt_2 + poly
    }
    return options[model], pars




def init_guess(x, y):
###### Same as in the old fitting.py
    y00 = (y[0] + y[-1]) / 2  # np.percentile(y, 10)
    s0 = (np.max(x) - np.min(x)) / 30
    A0 = (np.max(y) - y00) * s0
    x00 = (np.max(x) + np.min(x)) / 2
    print(np.array([A0, s0]), np.array([x00, y00]))
    return A0, s0, x00, y00


    
    
    
def linefit(wave, flux, predict_wl, linewidth, outputplot, model='GP', degree=0):
######### This function fits the lines
    model = selectModel(model, wave, flux, predict_wl, linewidth, degree=degree)
    init = model[0].eval(model[1], x=wave)
    out = model[0].fit(flux, model[1], x=wave)
    out.params.pretty_print()
    print("The reduced chi-square of the fit is {:.6f}".format(out.redchi))
    lineplot = (out.plot(xlabel="Wavelength", ylabel="Flux"))
    lineplot[0].savefig(str(outputplot)+".pdf")
    
    

