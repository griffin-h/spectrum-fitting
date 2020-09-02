#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from lightcurve_fitting.speccal import readspec
from corner import corner
from astropy.table import hstack, Table
from astropy.time import Time
from astropy.constants import c
import argparse
from matplotlib.gridspec import GridSpec
from fitting import MCgauss, gaussian


def closestpt(x, y, refx, refy, ax=None):
    if ax is None:
        ax = plt.gca()
    pixx, pixy = ax.transData.transform(list(zip(refx, refy))).T
    dist = ((x - pixx) ** 2 + (y - pixy) ** 2) ** 0.5
    i = np.argmin(dist)
    return i


class LinePlot:
    def __init__(self, line, refx, refy, ax2, ax3, profile, linewl, otherwl, corner=False):
        self.line = line
        self.refx = refx
        self.refy = refy
        self.press = False
        self.i0 = None
        self.i1 = None
        self.ax2 = ax2
        self.ax3 = ax3
        self.fits = []
        self.params = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        self.profile = profile
        self.linewl = linewl
        self.otherwl = otherwl
        self.corner = corner

    def connect(self):
        """connect to all the events we need"""
        self.cidpress = self.line.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.line.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.line.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.line.axes: return
        self.press = True
        self.i0 = closestpt(event.x, event.y, self.refx, self.refy, self.line.axes)
        self.line.set_xdata([self.refx[self.i0]] * 2)
        self.line.set_ydata([self.refy[self.i0]] * 2)
        for ax in self.ax2[:-1] + self.ax3[:-1]:
            ax.cla()
            plt.setp(ax.get_xticklabels(), visible=False)
        self.ax2[-1].cla()
        self.ax3[-1].cla()
        for l in self.fits:
            self.line.axes.lines.remove(l)
        self.fits = []
        self.line.figure.canvas.draw()

    def on_motion(self, event):
        """on motion we will move the line if the mouse is over us"""
        if not self.press: return
        x0 = self.line.get_xdata()
        y0 = self.line.get_ydata()
        x0[1] = event.xdata
        y0[1] = event.ydata
        self.line.set_xdata(x0)
        self.line.set_ydata(y0)
        self.line.figure.canvas.draw()

    def on_release(self, event):
        """on release we reset the press data"""
        self.press = False
        self.i1 = closestpt(event.x, event.y, self.refx, self.refy, self.line.axes)
        x0 = self.line.get_xdata()
        y0 = self.line.get_ydata()
        x0[1] = self.refx[self.i1]
        y0[1] = self.refy[self.i1]
        self.line.set_xdata(x0)
        self.line.set_ydata(y0)
        self.line.figure.canvas.draw()

        i0 = min(self.i0, self.i1)
        i1 = max(self.i0, self.i1)
        x = self.refx[i0:i1]
        y = self.refy[i0:i1]
        sampler, pos, fitfunc = MCgauss(x, y, profile=self.profile, linewl=self.linewl, otherwl=self.otherwl)
        for i in range(len(self.ax2)):
            self.ax2[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)
        sampler.reset()
        sampler.run_mcmc(pos, 1000)
        for i in range(len(self.ax3)):
            self.ax3[i].plot(sampler.chain[:, :, i].T, 'k', alpha=0.2)

        if self.profile == 'emis':
            amplitude, width, center, continuum = sampler.flatchain.T
            slope = 0.
        elif self.profile == 'bc' or self.profile == 'rc':
            amplitude, width, slope, center, continuum = sampler.flatchain.T
        elif self.profile == 'pcyg' or self.profile == 'two':
            amplitude, width, amplitude_1, center, continuum, center_1 = sampler.flatchain.T
            slope = 0.
        elif self.profile == 'pcygbc' or self.profile == 'twobc':
            amplitude, width, amplitude_1, slope, center, continuum, center_1 = sampler.flatchain.T
        else:
            raise ValueError(f'profile "{self.profile}" not recognized')

        y1 = y[:, np.newaxis]

        # subtract off the second emission line
        if 'two' in self.profile:
            y1 = y1 - gaussian(x[:, np.newaxis], amplitude_1, width, center_1, 0)
            self.fits += self.line.axes.plot(x, y1.mean(axis=-1), color='b')

        # subtract off the flat or linear continuum
        y1 = y1 - continuum - slope * (x[:, np.newaxis] - center)

        if 'pcyg' in self.profile:
            velocity = c * (1. - center_1 / center)  # absorption minimum
        else:
            velocity = 2. * np.sqrt(2. * np.log(2.)) * c * width / center  # FWHM of the emission line
        equivwidth = np.trapz(y1 / continuum, x, axis=0)
        flux = np.trapz(y1, x, axis=0)

        v = np.mean(velocity)
        dv = np.std(velocity)
        ew = np.mean(equivwidth)
        dew = np.std(equivwidth)
        f = np.mean(flux)
        df = np.std(flux)
        self.params = [v, dv, ew, dew, f, df]

        print('FWHM velocity = {:.3f} +/- {:.3f}'.format(v, dv))
        print('equivalent width = {:.3f} +/- {:.3f}'.format(ew, dew))
        print('flux = {:.3e} +/- {:.3e}'.format(f, df))

        ps = [sampler.flatchain[i] for i in np.random.choice(sampler.flatchain.shape[0], 100)]
        for p in ps:
            yfit = fitfunc(x, *p)
            l = self.line.axes.plot(x, yfit, color='k', alpha=0.05)
            self.fits.append(l[0])
            if 'two' in self.profile:
                if len(p) == 6:
                    A, s, rA, x0, y0, dx = p
                    m = 0
                elif len(p) == 7:
                    A, s, rA, m, x0, y0, dx = p
                yfitA = gaussian(x, A, s, x0, y0) - m * (x - x0)
                lA = self.line.axes.plot(x, yfitA, color='g', alpha=0.05)
                self.fits.append(lA[0])
                yfitB = gaussian(x, A * rA, s, x0 + dx, y0) - m * (x - x0)
                lB = self.line.axes.plot(x, yfitB, color='r', alpha=0.05)
                self.fits.append(lB[0])

        self.line.figure.canvas.draw()

        if self.corner:
            corner(sampler.flatchain, labels=['A', 's', 'm', 'x0', 'y0', 'dx'])

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.line.figure.canvas.mpl_disconnect(self.cidpress)
        self.line.figure.canvas.mpl_disconnect(self.cidrelease)
        self.line.figure.canvas.mpl_disconnect(self.cidmotion)


def labelaxes(*axes_sets):
    for ax in axes_sets:
        ax[0].set_ylabel('$A$')
        ax[1].set_ylabel('$\sigma$')
        ax[2].set_ylabel('$x_0$')
        ax[3].set_ylabel('$y_0$')
        ax[3].set_xlabel('MCMC Step')


def plot_results(t, ycol='flux'):
    plt.errorbar(t['phase'], t[ycol], t['d'+ycol], fmt='o')
    plt.xlabel('Phase')
    plt.ylabel(ycol.capitalize())
    plt.show()


def measure_specvels(spectra, profile, linewl, l2=0., viewwidth=20., corner=False):
    tparams = Table(names=('v', 'dv', 'EW', 'dEW', 'flux', 'dflux'))
    if profile == 'two':
        nparams = 6
        figsize = (5, 15)
    elif profile == 'pcyg':
        nparams = 6
        figsize = (5, 15)
    elif profile == 'emis':
        nparams = 4
        figsize = (5, 10)
    elif profile == 'bc' or profile == 'rc':
        nparams = 5
        figsize = (5, 12.5)
    elif profile == 'pcygbc':
        nparams = 7
        figsize = (5, 15)
    elif profile == 'twobc':
        nparams = 7
        figsize = (5, 15)
    else:
        raise ValueError('unrecognized profile type')
    hr = [4] + [1] * nparams
    gs = GridSpec(nparams + 1, 2, height_ratios=hr)
    for spec in spectra:
        good = ~np.isnan(spec['flux'])
        wl = spec['wl'][good]
        flux = spec['flux'][good]
        yshown = flux[(wl > linewl - viewwidth) & (wl < linewl + viewwidth)]
        if len(yshown) == 0:
            print('out of range for', spec['filename'], '(phase = {:+.1f})'.format(spec['phase']))
            tparams.add_row([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            continue
        f = plt.figure(figsize=figsize)
        ax1 = plt.subplot(gs[0:2])
        ax1.step(wl, flux)
        ax1.axvline(linewl, color='k', linestyle='-.', zorder=0)
        ax1.axis([linewl - viewwidth, linewl + viewwidth, np.min(yshown), np.max(yshown)])
        ax1.xaxis.tick_top()
        ax1.xaxis.set_ticks_position('both')

        l = plt.Line2D([], [])
        ax1.add_artist(l)

        ax2 = [plt.subplot(gs[2])]
        ax3 = [plt.subplot(gs[3])]
        for row in range(2, nparams + 1):
            ax2.append(plt.subplot(gs[2 * row], sharex=ax2[0]))
            ax3.append(plt.subplot(gs[2 * row + 1], sharex=ax3[0]))
        for ax in ax3:
            ax.yaxis.tick_right()
            ax.yaxis.set_ticks_position('both')
        for ax in ax2[:-1] + ax3[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        plt.subplots_adjust(left=0.11, right=0.89, top=0.97, bottom=0.03, hspace=0, wspace=0)

        dr = LinePlot(l, wl, flux, ax2, ax3, profile, linewl, l2, corner)
        dr.connect()
        plt.show()

        # cont = input('Press enter to accept these values or r to reject them.\n')
        # plt.clf()
        # if dr.corner:
        #     plt.figure(1)
        #     plt.clf()
        # if cont == 'r':
        #     print('fit rejected')
        #     tparams.add_row([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        # else:
        tparams.add_row(dr.params)

    for col in tparams.colnames:
        tparams[col].format = '%.3e'
    tout = hstack([spectra[['filename', 'date', 'telescope', 'instrument', 'phase']], tparams])
    tout[['filename', 'v', 'EW', 'flux']].pprint(max_width=-1, max_lines=-1)
    plot_results(tout)
    return tout


def read_spectra(filenames, redshift, refmjd=0.):
    wls = []
    fluxes = []
    dates = []
    telescopes = []
    instruments = []
    for fn in filenames:
        wl, flux, date, tel, inst = readspec(fn)
        wl /= (1 + redshift)  # correct to rest wavelength
        if wl[0] > wl[-1]:  # if the spectrum is stored backwards, flip it now
            wl = wl[::-1]
            flux = flux[::-1]
        wls.append(wl)
        fluxes.append(flux)
        dates.append(Time.now() if date is None else date)
        telescopes.append(tel)
        instruments.append(inst)
    if len(wls) and all([len(wl) == len(wls[0]) for wl in wls]):  # stupid astropy workaround
        filenames.append('fake')
        wls.append(np.array([]))
        fluxes.append(np.array([]))
        dates.append(Time.now())
        telescopes.append('fake')
        instruments.append('fake')
        spectra = Table([filenames, dates, telescopes, instruments, wls, fluxes],
                        names=['filename', 'date', 'telescope', 'instrument', 'wl', 'flux'],
                        dtype=[str, np.object, str, str, np.ndarray, np.ndarray])
        spectra = spectra[:-1]
    else:
        spectra = Table([filenames, dates, telescopes, instruments, wls, fluxes],
                        names=['filename', 'date', 'telescope', 'instrument', 'wl', 'flux'],
                        dtype=[str, np.object, str, str, np.ndarray, np.ndarray])
    spectra['phase'] = (spectra['date'].mjd - refmjd) / (1. + redshift)
    return spectra


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure line velocities and equivalent widths in spectra.')
    parser.add_argument('filenames', nargs='+', help='filenames of the spectra')
    parser.add_argument('-l', '--wavelength', type=float, default=6563., help="wavelength of line")
    parser.add_argument('--l2', type=float, default=0, help="wavelength of second line")
    parser.add_argument('-z', '--redshift', type=float, default=0., help="redshift to correct the spectra")
    parser.add_argument('-t', '--refmjd', type=float, default=0., help="reference MJD to calculate phase")
    parser.add_argument('-w', '--viewwidth', type=float, default=200., help="estimated width of the line")
    parser.add_argument('-p', '--profile', default='emis', choices=['two', 'pcyg', 'emis', 'bc', 'rc', 'pcygbc', 'twobc'],
                        help="emission line (emis) or P Cygni line (pcyg)?")
    parser.add_argument('-c', '--corner', action='store_true', help="show corner plot")
    args = parser.parse_args()

    spectra = read_spectra(args.filenames, args.redshift, refmjd=args.refmjd)
    tout = measure_specvels(spectra, args.profile, args.wavelength, args.l2, args.viewwidth, args.corner)
    savefile = input('Save results as... ')
    if savefile:
        tout.write(savefile, format='ascii.fixed_width_two_line')
