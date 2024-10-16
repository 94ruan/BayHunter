# #############################
#
# Copyright (C) 2024
# RUAN Yihuan   (ruan.yihuan.86s@st.kyoto-u.ac.jp)
#
#
# #############################

import numpy as np
from fraysum import run_bare
from IterDecon_bare import iterdecon
from pyraysum import Model, Geometry, Control

class IterRF(object):
    """
    """
    def __init__(self, obsx, ref):
        self.ref = ref
        self.obsx = obsx
        self._init_obsparams()
        self.modelparams = {'p': 0.06}
        # default parameters necessary for forward modeling
        # the dictionary can be updated by the user
        self.modelparams.update(
            {'gauss': 5.5,
             'odina_flag': 2,
             })
        
        self.keys = {'z': '%.2f',
                     'vpvs': '%.4f',
                     'vs': '%.4f',
                     'rho': '%.4f',
                     'n': '%d'}

    def _init_obsparams(self):
        """Extract parameters from observed x-data (time vector).

        fsamp = sampling frequency in Hz
        tshft = time shift by which the RF is shifted to the left
        nsamp = number of samples, must be 2**x
        """

        # get fsamp
        deltas = np.round((self.obsx[1:] - self.obsx[:-1]), 4)
        if np.unique(deltas).size == 1:
            self.dt = float(deltas[0])
            self.fsamp = 1. / self.dt
        else:
            raise ValueError("Target: %s. Sampling rate must be constant."
                             % self.ref)
        # get tshft
        self.tshft = -self.obsx[0] #重点关注需要改原RF计算时的时间轴戳

        # get nsamp
        ndata = self.obsx.size
        self.nsamp = 2.**int(np.ceil(np.log2(ndata * 2)))
        
    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def compute_rf(self, h, vp, vs, **params):
        """
        Compute RF using iterative deconvolution

        Parameters are:
        rho: Density 
        """
        
        rho = params.get('rho')
        mults = params.get('mults', 1)
        strike = params.get('strike', np.zeros(h.size))
        dip = params.get('dip', np.zeros(h.size))
        flag = params.get('flag', np.ones(h.size, dtype='int'))
        ani = params.get('ani', np.zeros(h.size))[0, :]
        trend = params.get('ani', np.zeros(h.size))[1, :]
        plunge = params.get('ani', np.zeros(h.size))[2, :]

        gauss = self.modelparams['gauss']
        odina_flag = self.modelparams['odina_flag']
        p = self.modelparams['p']
        npts = self.obsx.size
        baz = np.arange(0., 365., 5.)

        gaussF = gauss(npts, gauss, self.nsamp)
        geo = Geometry(baz, p)
        rc = Control(dt=self.dt, npts=npts, mults=mults, 
                     rot=odina_flag, shift=self.tshft)
        forward_list = np.zeros((len(baz), 2, npts), dtype=np.float16)
        vel_model = Model(h, rho, vp, vs,
                          strike=strike, dip=dip, flag=flag,
                          ani=ani, trend=trend, plunge=plunge)
        traces = run_bare(
                          vel_model.fthickn, vel_model.frho, vel_model.fvp,
                          vel_model.fvs, vel_model.fflag, vel_model.fani,
                          vel_model.ftrend, vel_model.fplunge, vel_model.fstrike,
                          vel_model.fdip, vel_model.nlay,
                          *geo.parameters,
                          *rc.parameters).transpose(2, 0, 1)
        # z = np.cumsum(h)
        # z = np.concatenate(([0], z[:-1]))
        time = np.arange(self.nsamp) / self.fsamp - self.tshft
        forward_list[:, :, :] = iterdecon(traces, baz, self.nsamp, gaussF,
                                          tshift=self.tshft, dt_bare= self.dt, nt_bare=npts)
        forward_list = np.concatenate((forward_list[:, 0, :], 
                                       forward_list[:, 1 ,:]), axis=1)
        # must be converted to float64
        # rfdata = forward_list.astype(float)

        return time[:self.obsx.size], forward_list

    # def validate(self, xmod, ymod):
    #     """Some condition that modeled data is valid. """
    #     if ymod.size == self.obsx.size:
    #         # xmod == xobs !!!
    #         return xmod, ymod
    #     else:
    #         return np.nan, np.nan

    def run_model(self, h, vp, vs, **params):

        assert h.size == vp.size == vs.size

        time, rf_list = self.compute_rf(h, vp, vs, **params)

        return time, rf_list
