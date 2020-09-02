import numpy as np
import emcee


def gaussian(x, A, s, x0, y0):
    return A / s * np.exp(-(x - x0) ** 2 / (2 * s ** 2)) + y0


def pcygni(x, A, s, A1, x0, y0, x1):
    return gaussian(x, A, s, x0, y0) - gaussian(x, A1, s, x1, 0)


def twogaussians(x, A, s, A1, x0, y0, x1):
    return gaussian(x, A, s, x0, y0) + gaussian(x, A1, s, x1, 0)


def gaussbc(x, A, s, m, x0, y0):
    return gaussian(x, A, s, x0, y0) - m * (x - x0)


def gaussrc(x, A, s, m, x0, y0):
    return gaussian(x, A, s, x0, y0) + m * (x - x0)


def pcygnibc(x, A, s, A1, m, x0, y0, x1):
    return pcygni(x, A, s, A1, x0, y0, x1) - m * (x - x0)


def twobc(x, A, s, A1, m, x0, y0, x1):
    return gaussian(x, A, s, x0, y0) + gaussian(x, A1, s, x1, 0) - m * (x - x0)


gaussian.param_names = ('amplitude', 'stddev', 'mean', 'intercept')
pcygni.param_names = ('amplitude', 'stddev', 'amplitude_1', 'mean', 'intercept', 'mean_1')
twogaussians.param_names = ('amplitude', 'stddev', 'amplitude_1', 'mean', 'intercept', 'mean_1')
gaussbc.param_names = ('amplitude', 'stddev', 'slope', 'mean', 'intercept')
gaussrc.param_names = ('amplitude', 'stddev', 'slope', 'mean', 'intercept')
pcygnibc.param_names = ('amplitude', 'stddev', 'amplitude_1', 'slope', 'mean', 'intercept', 'mean_1')
twobc.param_names = ('amplitude', 'stddev', 'amplitude_1', 'slope', 'mean', 'intercept', 'mean_1')


def init_guess_emis1(x, y):
    y00 = (y[0] + y[-1]) / 2  # np.percentile(y, 10)
    s0 = (np.max(x) - np.min(x)) / 30
    A0 = (np.max(y) - y00) * s0
    x00 = (np.max(x) + np.min(x)) / 2
    print(np.array([A0, s0]), np.array([x00, y00]))
    return np.array([A0, s0]), np.array([x00, y00])


def init_guess_emis(x, y, continuum_sign=0):
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    xavg = (x[0] + x[-1]) / 2.
    yavg = (y[0] + y[-1]) / 2.
    if continuum_sign:
        slope = dy / dx
    else:
        slope = 0
    continuum = slope * (x - xavg) + yavg
    i0 = np.argmax(y - continuum)
    x0 = x[i0]
    y0 = slope * (x0 - xavg) + yavg
    s = dx / 30
    A = (y[i0] - y0) * s
    if continuum_sign:
        print(np.array([A, s, continuum_sign * slope]), np.array([x0, y0]))
        return np.array([A, s, continuum_sign * slope]), np.array([x0, y0])
    else:
        print(np.array([A, s]), np.array([x0, y0]))
        return np.array([A, s]), np.array([x0, y0])


def init_guess_pcyg(x, y, bc=False):
    if bc:
        slope = (y[0] - y[-1]) / (x[-1] - x[0])
    else:
        slope = 0
    y00 = (y[0] + y[-1]) / 2  # np.mean(y)
    xctr = (x[0] + x[-1]) / 2
    cont = slope * (x - xctr)
    xem0 = x[np.argmax(y + cont)]  # np.percentile(x, 60)
    x_abs = x[x < xem0]
    y_abs = y[x < xem0]
    c_abs = cont[x < xem0]
    xab0 = x_abs[np.argmin(y_abs + c_abs)]  # np.percentile(x, 40)
    s0 = (xem0 - xab0) / 2  # (np.max(x) - np.min(x)) / 50
    A0 = (np.max(y + cont) - y00) * s0
    if bc:
        print(np.array([A0, s0, A0, slope]), np.array([xem0, y00, xab0]))
        return np.array([A0, s0, A0, slope]), np.array([xem0, y00, xab0])
    else:
        print(np.array([A0, s0, A0]), np.array([xem0, y00, xab0]))
        return np.array([A0, s0, A0]), np.array([xem0, y00, xab0])


def init_guess_two(x, y, bc=False):
    [A0, s0], [x00, y00] = init_guess_emis1(x, y)
    dx0 = 80.
    if bc:
        slope = (y[0] - y[-1]) / (x[-1] - x[0])
        print(np.array([A0, s0, A0, slope]), np.array([x00, y00, x00 + dx0]))
        return np.array([A0, s0, A0, slope]), np.array([x00, y00, x00 + dx0])
    else:
        print(np.array([A0, s0, A0]), np.array([x00, y00, x00 + dx0]))
        return np.array([A0, s0, A0]), np.array([x00, y00, x00 + dx0])


def MCgauss(x, y, perc_err=0.05, profile='emis', linewl=0, otherwl=0):
    dy = np.median(y) * perc_err  # assume 5% error
    if profile == 'emis':
        p0_mult, p0_add = init_guess_emis(x, y)
        unc_mult = np.array([7., 7.])
        unc_add = np.array([3 * p0_mult[1], p0_add[1]])
        fitfunc = gaussian
    elif profile == 'pcyg':
        p0_mult, p0_add = init_guess_pcyg(x, y)
        unc_mult = np.array([7., 7., 7.])
        unc_add = np.array([2 * p0_mult[1], p0_add[1], 2 * p0_mult[1]])
        fitfunc = pcygni
    elif profile == 'two':
        p0_mult, p0_add = init_guess_two(x, y)
        if linewl:
            p0_add[0] = linewl
        if otherwl:
            p0_add[2] = otherwl - linewl
        unc_mult = np.array([2., 2., 10.])
        unc_add = np.array([2 * p0_mult[1], p0_add[1], 2 * p0_mult[1]])
        fitfunc = twogaussians
    elif profile == 'bc':
        p0_mult, p0_add = init_guess_emis(x, y, continuum_sign=-1)
        unc_mult = np.array([5., 5., 2.])
        unc_add = np.array([30., p0_add[1]])
        fitfunc = gaussbc
    elif profile == 'rc':
        p0_mult, p0_add = init_guess_emis(x, y, continuum_sign=1)
        unc_mult = np.array([5., 5., 2.])
        unc_add = np.array([30., p0_add[1]])
        fitfunc = gaussrc
    elif profile == 'pcygbc':
        p0_mult, p0_add = init_guess_pcyg(x, y, bc=True)
        if linewl:
            p0_add[0] = linewl
        unc_mult = np.array([5., 5., 5., 2.])
        unc_add = np.array([20., p0_add[1], 20.])
        fitfunc = pcygnibc
    elif profile == 'twobc':
        p0_mult, p0_add = init_guess_two(x, y, bc=True)
        if linewl:
            p0_add[0] = linewl
        if otherwl:
            p0_add[2] = otherwl - linewl
        unc_mult = np.array([5., 5., 5., 2.])
        unc_add = np.array([20., p0_add[1], 20.])
        fitfunc = twobc

    nmult = len(p0_mult)

    def log_prior(p, nmult):
        p_mult, p_add = np.split(p, [nmult])
        if any(p_mult <= 0) or any(np.abs(np.log(p_mult / p0_mult)) >= np.log(unc_mult)) or any(
                np.abs(p_add - p0_add) >= unc_add):
            return -np.inf
        else:
            return -np.sum(np.log(p_mult))

    def log_likelihood(p, x, y, dy):
        y_model = fitfunc(x, *p)
        return -0.5 * np.sum(np.log(2 * np.pi * dy ** 2) + ((y - y_model) / dy) ** 2)

    def log_posterior(p, nmult, x, y, dy):
        return log_prior(p, nmult) + log_likelihood(p, x, y, dy)

    ndim = nmult + len(p0_add)
    nwalkers = 100

    starting_guesses = [p0_i * np.exp(np.random.uniform(-np.log(unc_i), np.log(unc_i), nwalkers)) for p0_i, unc_i in
                        zip(p0_mult, unc_mult)]
    starting_guesses += [p0_i + np.random.uniform(-unc_i, unc_i, nwalkers) for p0_i, unc_i in zip(p0_add, unc_add)]
    starting_guesses = np.array(starting_guesses).T

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[nmult, x, y, dy])
    pos, prob, state = sampler.run_mcmc(starting_guesses, 200)
    return sampler, pos, fitfunc
