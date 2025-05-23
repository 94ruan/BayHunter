# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import logging
import numpy as np
import matplotlib.pyplot as plt
logger = logging.getLogger()


class ObservedData(object):
    """
    The observed data object only consists of x and y.

    x = continuous and monotone increasing vector
    y = y(x)

    For iterative deconvolution, `y` consists of 73 traces,
    representing 0 to 360 degree with 5 degree interval.
    Each trace is constructed by radial and transverse component.

    """
    def __init__(self, x, y, yerr=None):
        self.x = x
        self.y = y
        self.yerr = yerr

        if self.yerr is None or np.any(yerr<=0.) or np.any(np.isnan(yerr)):
            self.yerr = np.ones(x.size) * np.nan


class ModeledData(object):
    """
    The modeled data object consists of x and y, which are initiated with nan,
    and will be computed during the inversion with the forward modeling tools.
    The plugins are python wrappers returning synthetic data, based on:
    RF: RFmini (Joachim Saul, GFZ, Posdam)
    SW: Surf96 (Rob Herrmann, St. Louis University, USA)

    You can easily update the plugin with your own code. Initiate the plugin
    with the necessary parameters and forward the instance to the
    update_plugin(instance) method. You can access this method through the
    SingleTarget object.

    The final method returning synthetic x and y data must be named
    self.run_model(h, vp, vs, rho, **kwargs). You can find a template with
    necessary plugin structure and method names in the defaults folder.
    Get inspired by the source code of the existing plugins.
    """
    def __init__(self, obsx, ref):
        rf_targets = ['prf', 'srf']
        swd_targets = ['rdispph', 'ldispph', 'rdispgr', 'ldispgr']

        if ref in rf_targets:
            from BayHunter.rfmini_modrf import RFminiModRF
            self.plugin = RFminiModRF(obsx, ref)
            self.xlabel = 'Time in s'

        elif ref in swd_targets:
            from BayHunter.surf96_modsw import SurfDisp
            self.plugin = SurfDisp(obsx, ref)
            self.xlabel = 'Period in s'

        elif ref in ['iterrf']:
            from BayHunter.iterrf import IterRF
            self.plugin = IterRF(obsx, ref)
            self.xlabel = 'Time in s'

        else:
            message = "Please provide a forward modeling plugin for your " + \
                "target.\nUse target.update_plugin(MyForwardClass())"
            logger.info(message)
            self.plugin = None
            self.xlabel = 'x'

        self.x = np.nan
        self.y = np.nan

    def update(self, plugin):
        self.plugin = plugin

    def calc_synth(self, h, vp, vs, **kwargs):
        """ Call forward modeling method of plugin."""
        rho = kwargs.pop('rho')
        
        self.x, self.y = self.plugin.run_model(h, vp, vs, rho=rho, **kwargs)


class Valuation(object):
    """
    Computation methods for likelihood and misfit are provided.
    The RMS misfit is only used for display in the terminal to get an estimate
    of the progress of the inversion.

    ONLY the likelihood is used for Bayesian inversion.
    """
    def __init__(self):
        self.corr_inv = None
        self.logcorr_det = None
        self.misfit = None
        self.likelihood = None

    @staticmethod
    def get_rms(yobs, ymod):
        """Return root mean square."""
        rms = np.sqrt(np.mean((ymod - yobs)**2))
        return rms

    @staticmethod
    def get_covariance_nocorr(sigma, size, yerr=None, corr=0):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) of 0.

        If there is no correlation between data points, the correlation matrix
        is represented by the diagonal.
        """
        c_inv = np.diag(np.ones(size)) / (sigma**2)
        logc_det = (2*size) * np.log(sigma)
        return c_inv, logc_det

    @staticmethod
    def get_covariance_nocorr_scalederr(sigma, size, yerr, corr=0):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) of 0.

        If there is no correlation between data points, the correlation matrix
        is represented by the diagonal. Errors are relatively scaled.
        """
        scaled_err = yerr / yerr.min()

        c_inv = np.diag(np.ones(size)) / (scaled_err * sigma**2)
        logc_det = (2*size) * np.log(sigma) + np.log(np.product(scaled_err)) 
        return c_inv, logc_det

    @staticmethod
    def get_corr_inv(corr, size):
        d = np.ones(size) + corr**2
        d[0] = d[-1] = 1
        e = np.ones(size-1) * -corr
        corr_inv = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
        return corr_inv

    def get_covariance_exp(self, corr, sigma, size, yerr=None):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) not equaling 0.

        The correlation between data points is represented by an EXPONENTIAL law.
        """
        c_inv = self.get_corr_inv(corr, size) / (sigma**2 * (1-corr**2))
        logc_det = (2*size) * np.log(sigma) + (size-1) * np.log(1-corr**2)

        return c_inv, logc_det

    def init_covariance_gauss(self, corr, size, rcond=None):
        idx = np.fromfunction(lambda i, j: (abs((i+j) - 2*i)),
                              (size, size))
        rmatrix = corr**(idx**2)

        if rcond is not None:
            self.corr_inv = np.linalg.pinv(rmatrix, rcond=rcond)
        else:
            self.corr_inv = np.linalg.inv(rmatrix)
        _, logdet = np.linalg.slogdet(rmatrix)
        self.logcorr_det = logdet

    def get_covariance_gauss(self, sigma, size, yerr=None, corr=None):
        """Return inverse and log-determinant of covariance matrix
        for a correlation (corr) not equaling 0.

        The correlation between data points is represented by a GAUSSIAN law.
        Consider this type of correlation if a gaussian filter was applied
        to compute RF. In this case, the inverse and log-determinant of the
        correlation matrix R is computed only once when initiating the chains.
        """
        c_inv = self.corr_inv / (sigma**2)
        logc_det = (2*size) * np.log(sigma) + self.logcorr_det
        return c_inv, logc_det

    @staticmethod
    def get_likelihood(yobs, ymod, c_inv, logc_det, ref, traceflag=False):
        """Return log-likelihood."""
        ydiff = ymod - yobs
        if ref=='iterrf':
            traceflag_arg = np.where(traceflag!=0)[0]
            ydiff = ydiff[traceflag_arg]
            halfsize = int(yobs.shape[-1] / 2)
            madist = ((ydiff[:, :halfsize]).dot(c_inv).dot(ydiff[:, :halfsize].T) + \
                            (ydiff[:, -halfsize:]).dot(c_inv).dot(ydiff[:, -halfsize:].T)).trace() / np.sum(traceflag)
        else:
            madist = (ydiff.T).dot(c_inv).dot(ydiff)  # Mahalanobis distance
        logL_part = -0.5 * (yobs.shape[-1] * np.log(2*np.pi) + logc_det)
        logL = logL_part - madist / 2.

        return logL


class SingleTarget(object):
    """A SingleTarget object gathers observed and modeled data,
    and the valuation methods. It provides methods to calculate misfit and
    likelihood, and also a plotting method. These can be used when initiating
    and testing your targets.
    """
    def __init__(self, x, y, ref, yerr=None, **kwarg):
        self.ref = ref
        self.obsdata = ObservedData(x=x, y=y, yerr=yerr)
        self.moddata = ModeledData(obsx=x, ref=ref)
        self.valuation = Valuation()

        if (ref == 'iterrf'):
            self.traceflag = kwarg.get('traceflag', np.ones(73))
            self.traceflag_arg = np.where(self.traceflag!=0)[0]
            self.traceweight = kwarg.get('traceweight', np.ones(73))
        logger.info("Initiated target: %s (ref: %s)"
                    % (self.__class__.__name__, self.ref))

    def update_plugin(self, plugin):
        self.moddata.update(plugin)

    def _moddata_valid(self):
        if not type(self.moddata.x) == np.ndarray:
            return False
        if not len(self.obsdata.x) == len(self.moddata.x):
            return False
        if not np.sum(self.obsdata.x - self.moddata.x) <= 1e-5:
            return False
        if not len(self.obsdata.y) == len(self.moddata.y):
            return False

        return True

    def calc_misfit(self):
        if not self._moddata_valid():
            self.valuation.misfit = 1e15
            return

        if self.ref=='iterrf':
            self.valuation.misfit = self.valuation.get_rms(
                self.obsdata.y[self.traceflag_arg], self.moddata.y[self.traceflag_arg])
        else:
            self.valuation.misfit = self.valuation.get_rms(
                self.obsdata.y, self.moddata.y)

    def calc_likelihood(self, c_inv, logc_det):
        if not self._moddata_valid():
            self.valuation.likelihood = -1e15
            return

        self.valuation.likelihood = self.valuation.get_likelihood(
            self.obsdata.y, self.moddata.y, c_inv, logc_det, self.ref, self.traceflag)

    def plot(self, ax=None, mod=True):
        if ax is None:
            fig, ax = plt.subplots()

        # ax.plot(self.obsdata.x, self.obsdata.y, label='obs',
        #         marker='x', ms=1, color='blue', lw=0.8, zorder=1000)
        yobs = np.average(self.obsdata.y[self.traceflag_arg], axis=0)[:len(self.obsdata.x)] if self.obsdata.y.ndim != 1 else self.obsdata.y
        ax.errorbar(self.obsdata.x, yobs, yerr=self.obsdata.yerr,
                    label='obs', marker='x', ms=1, color='blue', lw=0.8,
                    elinewidth=0.7, zorder=1000)
        if mod:
            ymod = np.average(self.moddata.y[self.traceflag_arg], axis=0)[:len(self.obsdata.x)] if self.moddata.y.ndim != 1 else self.moddata.y
            ax.plot(self.moddata.x, ymod, label='mod',
                    marker='o',  ms=1, color='red', lw=0.7, alpha=0.5)

        ax.set_ylabel(self.ref)
        ax.set_xlabel(self.moddata.xlabel)

        return ax


class RayleighDispersionPhase(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None):
        ref = 'rdispph'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class RayleighDispersionGroup(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None):
        ref = 'rdispgr'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class LoveDispersionPhase(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None):
        ref = 'ldispph'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class LoveDispersionGroup(SingleTarget):
    noiseref = 'swd'

    def __init__(self, x, y, yerr=None):
        ref = 'ldispgr'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class PReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, yerr=None):
        ref = 'prf'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)


class SReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, yerr=None):
        ref = 'srf'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr)

class IterReceiverFunction(SingleTarget):
    noiseref = 'rf'

    def __init__(self, x, y, yerr=None, **kwarg):
        ref = 'iterrf'
        SingleTarget.__init__(self, x, y, ref, yerr=yerr, **kwarg) # traceflag should in a format of `[0, 1, 1, 1, 0, 1, 1]`
        
class JointTarget(object):
    """A JointTarget object contains a list of SingleTargets and is responsible
    for computing the joint likelihood, given all model parameters."""
    def __init__(self, targets):
        self.targets = targets  # list of SingleTargets
        self.ntargets = len(targets)

    def get_misfits(self):
        """Compute misfit by summing target misfits.
        Keep targets' individual misfits for comparison purposes."""
        misfits = [target.valuation.misfit for target in self.targets]
        jointmisfit = np.sum(misfits)
        return np.concatenate((misfits, [jointmisfit]))

    def evaluate(self, h, vp, vs, noise, **kwargs):
        """This evaluation method basically evaluates the given model.
        It computes the jointmisfit, and more important the jointlikelihoods.
        The jointlikelihood (here called the proposallikelihood) is the sum
        of the log-likelihoods from each target."""
        rho = kwargs.pop('rho', 
                         1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4\
                            + 0.000103*vp**5)
        # ani = kwargs.get('ani', None)
        # flag = kwargs.get('flag', None)

        logL = 0
        for n, target in enumerate(self.targets):
            if target.ref == 'iterrf':
                kwargs['traceflag_arg'] = target.traceflag_arg
            target.moddata.calc_synth(h=h, vp=vp, vs=vs, rho=rho, **kwargs)
            if not target._moddata_valid():
                self.proposallikelihood = -1e15
                self.proposalmisfits = [1e15]*(self.ntargets+1)
                return
            target.calc_misfit()
            size = target.obsdata.y.shape[-1]
            yerr = target.obsdata.yerr

            ydiff = target.moddata.y - target.obsdata.y
            if target.ref == 'iterrf':
                ydiff = ydiff[target.traceflag_arg]
                halfsize = int(size / 2)
                corr, sigma = noise[2*n:2*n+2]
                c_inv, logc_det = target.get_covariance(
                    sigma=sigma, size=halfsize, yerr=yerr, corr=corr)
                # r_madist = ((ydiff[:, :halfsize]).dot(c_inv).dot(ydiff[:, :halfsize].T)).trace() / 2.0
                # t_madist = ((ydiff[:, -halfsize:]).dot(c_inv).dot(ydiff[:, -halfsize:].T)).trace()
                # madist = (r_madist+ t_madist) / np.sum(target.traceflag) #TOTALLY 73 TRACES
                # logL_part = -0.5 * (halfsize * np.log(2*np.pi) + logc_det)

                # 计算加权残差
                weighted_ydiff_r = ydiff[:, :halfsize] * np.sqrt(target.traceweight[target.traceflag_arg, None])  # 加权前半部分
                weighted_ydiff_t = ydiff[:, -halfsize:] * np.sqrt(target.traceweight[target.traceflag_arg, None])  # 加权后半部分
                # 计算加权后的马氏距离
                r_madist = (weighted_ydiff_r.dot(c_inv).dot(weighted_ydiff_r.T)).trace()
                t_madist = (weighted_ydiff_t.dot(c_inv).dot(weighted_ydiff_t.T)).trace()
                # 归一化（按总权重调整）
                madist = (r_madist + t_madist) / np.sum(target.traceweight[target.traceflag_arg])  # 仅计算有效 trace 的总权重
            else:
                corr, sigma = noise[2*n:2*n+2]
                c_inv, logc_det = target.get_covariance(
                    sigma=sigma, size=size, yerr=yerr, corr=corr)
                madist = (ydiff.T).dot(c_inv).dot(ydiff)

            logL_part = -0.5 * (size * np.log(2*np.pi) + logc_det)
            logL_target = (logL_part - madist / 2.)
            # print(f"logL_part of {target.ref}={logL_part}, madist of {target.ref}={madist}")
            logL += logL_target
        # print('\n')
        self.proposallikelihood = logL
        self.proposalmisfits = self.get_misfits()
        # print(f'madist: [{madist}]; misfit: [{self.proposalmisfits}]; like: [{self.proposallikelihood}]')

    def plot_obsdata(self, ax=None, mod=False):
        """Return subplot of all targets."""
        if len(self.targets) == 1:
            if ax is None:
                fig, ax = plt.subplots(figsize=(7, 3.2))
            else:
                fig = ax.figure

            ax = self.targets[0].plot(ax=ax, mod=mod)
            ax.legend()

        else:
            if ax is None:
                fig, ax = plt.subplots(self.ntargets,
                                       figsize=(6, 3.2*self.ntargets))
            else:
                fig = ax[0].figure

            for i, target in enumerate(self.targets):
                ax[i] = target.plot(ax=ax[i], mod=mod)

            han, lab = ax[0].get_legend_handles_labels()
            ax[0].legend(han, lab)

        return fig, ax
