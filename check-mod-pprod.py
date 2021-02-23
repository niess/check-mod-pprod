#! /usr/bin/env python3

'''
This script is an attempt to reproduce the results of Yanez et al., 2019
(https://arxiv.org/pdf/1909.08365.pdf). Currently only the BESS-TeV data
are considered.
'''

import collections
import matplotlib.pyplot as plot
import numpy
import os

from MCEq.core import config, MCEqRun
import crflux.models as crf


# Load and format the BESS-TeV data
ExperimentalData = collections.namedtuple(
    'ExperimentalData', ('energy', 'flux', 'dflux'))

def load_bess(path):
    data = numpy.loadtxt(path, comments='#')
    m = 0.10566                     # Muon mass (GeV/c^2)
    p = data[:,2]                   # Momentum (GeV)
    e = numpy.sqrt(p**2 + m**2) - m # Kinetic energy (GeV)
    j = (e + m) / p                 # Jacobian factor for going from dphi / dp
                                    #   to dphi / dE (almost 1 ...)
    f = (data[:,3] + data[:,7]) * j # Flux (1/(GeV m^2 s sr))

    df = numpy.sqrt(data[:,4]**2 + data[:,5]**2 + # Flux uncertainty
                    data[:,8]**2 + data[:,9]**2) * j

    return ExperimentalData(e, f * 1E-04, df * 1E-04)

bess = load_bess('BESS_TEV.txt')


# Simulate the flux using MCEq
mceq = MCEqRun(
    interaction_model='SIBYLL23C',
    primary_model = (crf.GlobalSplineFitBeta, None),
    density_model = ('MSIS00', ('Tokyo', 'October')),
    theta_deg = 0
)

cos_theta = 0.95
theta = numpy.arccos(cos_theta) * 180 / numpy.pi
mceq.set_theta_deg(theta)

altitude = numpy.array((30.,))
X_grid = mceq.density_model.h2X(altitude * 1E+02)

def weight(xmat, egrid, name, c):
    return (1 + c) * numpy.ones_like(xmat)

mceq.set_mod_pprod(2212,  211, weight, ('a', 0.141)) # Coefficients taken
mceq.set_mod_pprod(2212, -211, weight, ('a', 0.116)) # from table 2 of Yanez et
mceq.set_mod_pprod(2212,  321, weight, ('a', 0.402)) # al.
mceq.set_mod_pprod(2212, -321, weight, ('a', 0.583))
mceq.regenerate_matrices(skip_decay_matrix=True)

mceq.solve(int_grid=X_grid)

energy = mceq.e_grid
flux = mceq.get_solution('mu-', grid_idx=0)
flux += mceq.get_solution('mu+', grid_idx=0)


# Plot the result
plot.figure()
plot.semilogx(energy, energy**3 * flux, 'k--')
plot.errorbar(bess.energy, bess.energy**3 * bess.flux,
    yerr=bess.energy**3 * bess.dflux, fmt='bo', label='BESS-TeV')
plot.xlabel('energy [GeV]')
plot.ylabel('$dN_\\mu/dE (E/\\mathrm{GeV})^3$ [GeV$^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]')
plot.axis((3, 3E+03, 0, 0.5))
plot.savefig('flux.pdf')

plot.show()
