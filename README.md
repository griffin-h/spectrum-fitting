# Spectrum Fitting

Here is the basic syntax of the script:
`specvels.py spectrum1.txt spectrum2.fits -l 656 -p emis`
This will attempt to fit an emission line (`emis`) the Hα line (at 656 nm) in the two spectra you provide.
To see other line profiles and other optional parameters, run `specvels.py -h`.

When the script is run, a window will pop up showing the line to fit.
You need to provide an initial guess for the fit parameters.
To do this, click and drag on the window to define the continuum of the spectrum (the baseline on top of which the line forms).
When you release, the script will run an MCMC fit and display the results. This may take a few seconds.
The lower panels show the traces of the parameter values during (left) and after (right) burn-in.
If you like the fit, close the window to continue on to the next spectrum.
If you don't like the fit, you can try again by clicking and dragging on the same window.
After all the spectra have been fit, you can provide a filename to which to save the resulting measurements.
 