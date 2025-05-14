# #############################
#
# Copyright (C) 2024
# RUAN Yihuan   (ruan.yihuan.86s@st.kyoto-u.ac.jp)
#
#
# #############################

import numpy as np
import time
import os
import gc
import psutil
import threading
from fraysum import run_bare_mcmc
from BayHunter.IterDecon_bare import gauss_filter#, iterdecon
from iterdecon_cython import iterdecon
from iterdecon_cython import ThreadLocalStorage
from pyraysum import Model, Geometry, Control
from .tls_helper import get_tls
from ctypes import POINTER, c_float, c_int, c_bool
from scipy.fft import fft, ifft

class IterRF(object):
    """
    """
    def __init__(self, obsx, ref):
        print(f"[PID {os.getpid()}] Initializing IterRF")
        self.ref = ref
        self.obsx = obsx
        self._init_obsparams()
        self.modelparams = {
            'p': 0.06,
            'gauss': 5.5,
            'odina_flag': 2,
        }
        # default parameters necessary for forward modeling
        # the dictionary can be updated by the user
        
        self.keys = {'z': '%.2f',
                     'vpvs': '%.4f',
                     'vs': '%.4f',
                     'rho': '%.4f',
                     'n': '%d'}
        
        # 线程本地存储
        # self._tls = get_tls(self.nsamp)
        
        # 预计算不变参数
        self._precompute_constants()
        
        self._setup_optimized_buffers()  # 替换原有内存池

    def _precompute_constants(self):
        """预计算不变参数"""
        self._time_axis = np.arange(self.nsamp) / self.fsamp - self.tshft
        self._baz_template = np.arange(0., 365., 5.)  # 方位角模板
        # 高斯滤波器预计算
        self.gaussF = np.asarray(
            gauss_filter(self.dt, self.modelparams['gauss'], self.nsamp),
            dtype=np.complex128, order='C'
        )
        self.gaussF_nor = np.asarray(
            fft(ifft(self.gaussF).real / np.max(np.abs(ifft(self.gaussF).real)) * 1),
            dtype=np.complex128, order='C'
        )
        self.realdata = self.modelparams.get('realdata', False)
        
    def _setup_optimized_buffers(self):
        """设置优化的内存缓冲区"""
        # 模型参数缓冲区
        self._model_buf = {
            'thick': np.zeros(10, dtype=np.float32, order='F'),
            'rho': np.zeros(10, dtype=np.float32, order='F'),
            'vp': np.zeros(10, dtype=np.float32, order='F'),
            'vs': np.zeros(10, dtype=np.float32, order='F'),
            'strike': np.zeros(10, dtype=np.float32, order='F'),
            'dip': np.zeros(10, dtype=np.float32, order='F'),
            'flag': np.zeros(10, dtype=np.int32, order='F'),
            'ani': np.zeros((3, 10), dtype=np.float32, order='F')
        }
        
        # 输出缓冲区 (保持Fortran顺序)
        self._traces_buf32 = np.zeros((3, 3000, 73), dtype=np.float32, order='F')
        self._traces_buf32 = np.asfortranarray(self._traces_buf32)
        self._traces_buf = np.zeros((3, 3000, 73), dtype=np.float64, order='C')

        # 前向结果缓冲区
        self._forward_buf = np.zeros((73, 2, self.nsamp), dtype=np.float32, order='C')

    def _prepare_model(self, h, vp, vs, params):
        """修改为直接操作_model_buf"""
        nlay = len(h)
        self._model_buf['thick'][:nlay] = h * 1000
        self._model_buf['vp'][:nlay] = vp * 1000
        self._model_buf['vs'][:nlay] = vs * 1000
        self._model_buf['rho'][:nlay] = params.get('rho', vp*0.32 + 0.77)
        self._model_buf['strike'][:nlay] = params.get('strike', np.zeros(nlay))
        self._model_buf['dip'][:nlay] = params.get('dip', np.zeros(nlay))
        if self.realdata and (nlay>2):
            self._model_buf['strike'][nlay-1] = 206
            self._model_buf['strike'][nlay-2] = 206
            self._model_buf['dip'][nlay-1] = 10
            self._model_buf['dip'][nlay-2] = 10
        self._model_buf['flag'][:nlay] = params.get('flag', np.ones(nlay, dtype=int))
        
        ani = params.get('ani', np.zeros((3, nlay)))
        self._model_buf['ani'][:, :nlay] = ani

        return Model(
            self._model_buf['thick'][:nlay],
            self._model_buf['rho'][:nlay],
            self._model_buf['vp'][:nlay],
            self._model_buf['vs'][:nlay],
            strike=self._model_buf['strike'][:nlay],
            dip=self._model_buf['dip'][:nlay],
            flag=self._model_buf['flag'][:nlay],
            ani=self._model_buf['ani'][0, :nlay],
            trend=self._model_buf['ani'][1, :nlay],
            plunge=self._model_buf['ani'][2, :nlay]
        ), nlay
        
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # 删除无法序列化的成员
    #     if '_tls' in state:
    #         del state['_tls']
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # 重建不可pickle的对象
    #     # 重建 _tls（如果你知道 nsamp）
    #     self._tls = get_tls(self.nsamp)
        
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
        self.tshft = -self.obsx[0] #重点关注需要改原RF计算时的时间轴戳（为正数）

        # get nsamp
        ndata = self.obsx.size
        self.nsamp = int(2.**int(np.ceil(np.log2(ndata * 2))))
        print(self.nsamp)
        
    def set_modelparams(self, **mparams):
        self.modelparams.update(mparams)

    def compute_rf(self, h, vp, vs, **params):
        """
        Compute RF using iterative deconvolution

        Parameters are:
        rho: Density 
        """
        
        # 准备参数
        traceflag_arg = params.get('traceflag_arg', np.arange(73))
        baz = np.asarray(self._baz_template[traceflag_arg], dtype=np.float64, order='C')

        # 模型准备
        # t0 = time.perf_counter()
        vel_model, nlay = self._prepare_model(h, vp, vs, params)
        geo = Geometry(baz, self.modelparams['p'], maxtr=73)
        rc = Control(
            dt=self.dt, npts=self.nsamp,
            mults=params.get('mults', 2),
            rot=self.modelparams['odina_flag'],
            shift=self.tshft
        )
        # t1 = time.perf_counter()

        # 执行计算 (直接操作内存)
        self._traces_buf32.fill(0)
        run_bare_mcmc(
            vel_model.fthickn, vel_model.frho, vel_model.fvp,
            vel_model.fvs, vel_model.fflag, vel_model.fani,
            vel_model.ftrend, vel_model.fplunge, vel_model.fstrike,
            vel_model.fdip, vel_model.nlay,
            *geo.parameters,
            *rc.parameters,
            self._traces_buf32  # traces
        )
        # t2 = time.perf_counter()

        # 反卷积 (使用预分配缓冲区)
        # self._forward_buf.fill(0)
        # self._traces_buf = self._traces_buf32.astype(np.float64, order='C')
        # self._forward_buf[traceflag_arg, :, :] = iterdecon(
        #     self._traces_buf, baz, self.nsamp, 
        #     self.gaussF, self.modelparams['odina_flag'],
        #     tshift=self.tshft, dt_bare=self.dt, 
        #     tls=get_tls(self.nsamp), nused=len(baz)
        # )

        self._forward_buf.fill(0)
        self._traces_buf = self._traces_buf32.astype(np.float64, order='C')
        self._forward_buf[traceflag_arg, :, :] = iterdecon(
            self._traces_buf, baz, self.nsamp, 
            self.gaussF, 
            # self.gaussF_nor, 
            self.modelparams['odina_flag'],
            tshift=self.tshft, dt_bare=self.dt, 
            tls=get_tls(self.nsamp), nused=len(baz)
        )

        # self._forward_buf.fill(0)
        # self._traces_buf = self._traces_buf32.astype(np.float64, order='C').transpose(2, 0, 1)
        # self._forward_buf[traceflag_arg, :, :] = iterdecon(
        #     self._traces_buf, baz, self.nsamp, 
        #     self.gaussF, self.modelparams['odina_flag'],
        #     tshift=self.tshft, dt_bare=self.dt
        # )
        # t3 = time.perf_counter()
        # 性能日志
        # print(f"[Timing] Params: {(t1-t0)*1000:.2f}ms | "
        #       f"Waveform: {(t2-t1)*1000:.2f}ms | "
        #       f"Deconv: {(t3-t2)*1000:.2f}ms")
        
        # 组装结果（零拷贝视图）
        valid_len = len(self._time_axis[self._time_axis <= self.obsx[-1]])
        result = np.empty((73, 2*valid_len), dtype=np.float32)
        result[:, :valid_len] = self._forward_buf[:, 0, :valid_len]
        result[:, valid_len:] = self._forward_buf[:, 1, :valid_len]

        return self._time_axis[:valid_len], result
        
    # def validate(self, xmod, ymod):
    #     """Some condition that modeled data is valid. """
    #     if ymod.size == self.obsx.size:
    #         # xmod == xobs !!!
    #         return xmod, ymod
    #     else:
    #         return np.nan, np.nan

    def run_model(self, h, vp, vs, **params):
        assert h.size == vp.size == vs.size

        time_axis, rf_list = self.compute_rf(h, vp, vs, **params)

        return time_axis, rf_list