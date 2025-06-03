# #############################
#
# Copyright (C) 2018
# Jennifer Dreiling   (dreiling@gfz-potsdam.de)
#
#
# #############################

import copy
import time
import numpy as np
import os.path as op
from fractions import Fraction

from BayHunter import Model, ModelMatrix
from BayHunter import utils

import logging
logger = logging.getLogger()


PAR_MAP = {'vsmod': 0, 'zvmod': 1, 'birth': 2, 'death': 2,
           'noise': 3, 'vpvs': 4, 'ani':5, 'trend':6, 'plunge':7}


class SingleChain(object):

    def __init__(self, targets, chainidx=0, initparams={}, modelpriors={},
                 sharedmodels=None, sharedmisfits=None, sharedlikes=None,
                 sharednoise=None, sharedvpvs=None, sharedani=None, random_seed=None):
        self.chainidx = chainidx
        self.rstate = np.random.RandomState(random_seed)

        defaults = utils.get_path('defaults.ini')
        self.priors, self.initparams = utils.load_params(defaults)
        self.initparams.update(initparams)
        self.priors.update(modelpriors)
        self.dv = (self.priors['vs'][1] - self.priors['vs'][0])
        if type(self.priors['vpvs']) != float:
            self.dvpvs = (self.priors['vpvs'][1] - self.priors['vpvs'][0])

        self.nchains = self.initparams['nchains']
        self.station = self.initparams['station']

        # set targets and inversion specific parameters
        self.targets = targets
        self.ani_flag = False
        ref_list = [target.ref for target in self.targets.targets]
        if 'iterrf' in ref_list:
            self.ani_flag = True
            self.dani = (self.priors['anistr'][1] - self.priors['anistr'][0])
            self.dtr = (self.priors['anitre'][1] - self.priors['anitre'][0])
            self.dplu = (self.priors['aniplu'][1] - self.priors['aniplu'][0])

        # set parameters
        self.iter_phase1 = int(self.initparams['iter_burnin'])
        self.iter_phase2 = int(self.initparams['iter_main'])
        self.iterations = self.iter_phase1 + self.iter_phase2
        self.iiter = -self.iter_phase1
        self.lastmoditer = self.iiter

        self.propdist = np.array(self.initparams['propdist'])
        self.acceptance = self.initparams['acceptance']
        self.thickmin = self.initparams['thickmin']
        self.maxlayers = int(self.priors['layers'][1]) + 1

        self.lowvelperc = self.initparams['lvz']
        self.highvelperc = self.initparams['hvz']
        self.mantle = self.priors['mantle']

        # chain models
        self._init_chainarrays(sharedmodels, sharedmisfits, sharedlikes,
                               sharednoise, sharedvpvs, sharedani)

        # init model and values
        self._init_model_and_currentvalues()


# init model and misfit / likelihood

    def _init_model_and_currentvalues(self):
        self.init_flag = True # for adjusting the anisotropic parameter when model changed
        self.currentvpvs = self.draw_initvpvs()
        imodel = self.draw_initmodel()
        inoise, corrfix = self.draw_initnoiseparams()
        
        self.set_target_covariance(corrfix[::2], inoise[::2], self.initparams['rcond'])

        vp, vs, h = Model.get_vp_vs_h(imodel, self.currentvpvs, self.mantle)

        if self.ani_flag:
            iani = self.draw_initani(imodel)
        else:
            iani = np.zeros((3, h.size))
            self.tempaniflag = np.ones(h.size)
        self.init_flag = False

        self.targets.evaluate(h=h, vp=vp, vs=vs, noise=inoise, ani=iani, flag=self.tempaniflag)

        logger.debug((vs, h))

        self.n = 0  # accepted models counter
        self.accept_as_currentmodel(imodel, inoise, self.currentvpvs, ani=iani, flag=self.tempaniflag)
        self.append_currentmodel()

    def draw_initmodel(self):
        while True:
            keys = self.priors.keys()
            zmin, zmax = self.priors['z']
            vsmin, vsmax = self.priors['vs']
            zpri = np.atleast_1d(self.priors['zpri'])

            if self.priors['fixed']:
                self.fixzmax = np.max(self.priors['fixeddep'])
                fix_z_vnoi = utils.calculate_layer_boundaries(self.priors['fixeddep'])
                zmin = np.max(self.fixzmax)
                fix_vs = self.priors['fixedvel']
                if np.min(fix_vs) < vsmin:
                    self.priors['vs'][0] = np.min(fix_vs)

            # + half space
            layers = zpri.size + 1  if (zpri is not None and zpri.size > self.priors['layers'][0]) else self.priors['layers'][0] + 1 

            vs = np.sort(self.rstate.uniform(low=vsmin, high=vsmax, size=layers))

            if self.priors['fixmohoparam'] is not None:
                vs[-1] = self.priors['fixmohoparam'][0] #+ self.rstate.normal(0, 0.01)

            if (self.priors['mohoest'] is not None and layers > 1):
                mean, std = self.priors['mohoest']
                moho = self.rstate.normal(loc=mean, scale=std)
                tmp_z = self.rstate.uniform(1, np.min([5, moho]))  # 1-5

                z_vnoi = np.concatenate((
                        [moho-tmp_z, moho+tmp_z],
                        self.rstate.uniform(low=zmin, high=zmax, size=(layers - 2))
                        )) if (layers - 2) > 0 else [moho-tmp_z, moho+tmp_z]

            elif (self.priors['zpri'] is not None):
                std = min(self.priors['zpri_std'], 0.5 * np.min(zpri))
                z_layers = np.sort(self.rstate.normal(loc=zpri, scale=std))
                tmp_z_vnoi = utils.calculate_layer_boundaries(z_layers, 
                            z_vnoi_pre=fix_z_vnoi) if self.priors['fixed'] else utils.calculate_layer_boundaries(z_layers)
                z_vnoi = np.concatenate((
                    tmp_z_vnoi,
                    self.rstate.uniform(low=zmin, high=zmax, size=(layers - len(tmp_z_vnoi)))
                )) if len(tmp_z_vnoi) < layers else tmp_z_vnoi

            else:
                z_vnoi = self.rstate.uniform(low=zmin, high=zmax, size=layers)

            z_vnoi.sort()

            model = np.concatenate((
                (fix_vs if self.priors['fixed'] else np.array([])),
                vs,
                (fix_z_vnoi[:-1] if self.priors['fixed'] else np.array([])),
                z_vnoi
            ))

            if self.priors['fixvpvs']:
                self.currentvpvs = self.fix_vpvs(self.currentvpvs, model[int(len(model)/2):])
            print('Initializing')
            if self._validmodel(model):
                return model

    def fix_vpvs(self, vpvs_in, z_vnoi_in):
        vpvs = np.atleast_1d(vpvs_in)
        z_vnoi = np.atleast_1d(z_vnoi_in)
        for i in range(len(z_vnoi) - 1):  # 遍历除最后一层外的所有层
            depth = z_vnoi[i]
            if depth < 5:  # 0-5km
                vpvs[i] = 1.85
            elif depth < 30:  # 5-30km
                vpvs[i] = 1.68
            else:
                vpvs[i] = 1.73
        return vpvs

    def draw_initnoiseparams(self):
        # for each target the noiseparams are (corr and sigma)
        noiserefs = ['noise_corr', 'noise_sigma']
        init_noise = np.ones(len(self.targets.targets)*2) * np.nan
        corrfix = np.zeros(len(self.targets.targets)*2, dtype=bool)

        self.noisepriors = []
        for i, target in enumerate(self.targets.targets):
            for j, noiseref in enumerate(noiserefs):
                idx = (2*i)+j
                noiseprior = self.priors[target.noiseref + noiseref]

                if type(noiseprior) in [int, float, np.float64]:
                    corrfix[idx] = True
                    init_noise[idx] = noiseprior
                else:
                    init_noise[idx] = self.rstate.uniform(
                        low=noiseprior[0], high=noiseprior[1])

                self.noisepriors.append(noiseprior)

        self.noiseinds = np.where(corrfix == 0)[0]
        if len(self.noiseinds) == 0:
            logger.warning('All your noise parameters are fixed. On Purpose?')

        return init_noise, corrfix

    def draw_initvpvs(self):
        if type(self.priors['vpvs']) == float:
            return self.priors['vpvs']

        zpri = np.atleast_1d(self.priors['zpri'])
        if (zpri is not None and zpri.size>self.priors['layers'][0]):
            layers = zpri.size + 1  # half space
        else:
            layers = self.priors['layers'][0] + 1  # half space

        vpvsmin, vpvsmax = self.priors['vpvs']
        vpvs = self.rstate.uniform(low=vpvsmin, high=vpvsmax, size=layers)

        if self.priors['fixmohoparam'] is not None:
            vpvs[-1] = self.priors['fixmohoparam'][1] #+ self.rstate.normal(0, 0.01)

        if self.priors['fixed']:
            self.fixlayers = len(np.atleast_1d(self.priors['fixeddep']))
            temp_fix_vpvs = np.repeat(1.9, self.fixlayers)
            vpvs = np.concatenate((temp_fix_vpvs, vpvs))
        else:
            self.fixlayers = 0
        return vpvs

    def draw_initani(self, imodel):
        n = int(imodel.size / 2)
        z_vnoi = imodel[-n:]

        z_ani_flag = np.where(
            (z_vnoi>self.priors['anilim'][0]) &
            (z_vnoi<self.priors['anilim'][1]), 0, 1
            )
        z_ani_flag[-1] = 1 # if including last layer

        self.tempaniflag = z_ani_flag 
        z_ani_ind = np.where(z_ani_flag == 0)[0] # the arguments/indexes within anisotropic zone of CURRENT model
        
        anistrmin, anistrmax = self.priors['anistr']
        anitremin, anitremax = int(round(self.priors['anitre'][0])), int(round(self.priors['anitre'][1]))
        aniplumin, aniplumax = int(round(self.priors['aniplu'][0])), int(round(self.priors['aniplu'][1]))
        ani = np.zeros((3, n))
        ani[0, z_ani_ind] = self.rstate.uniform(low=anistrmin, high=anistrmax, size=len(z_ani_ind)) # Ani strength
        ani[1, z_ani_ind] = self.rstate.randint(low=anitremin, high=anitremax+1, size=len(z_ani_ind)) # Trend
        ani[2, z_ani_ind] = self.rstate.randint(low=aniplumin, high=aniplumax+1, size=len(z_ani_ind)) # Plunge
        # 考虑如何在后续的迭代中，限制参数以及aniflag的变化范围

        if (self.priors['fixedani'] is not None):
            ani[:, :self.fixlayers] = self.priors['fixedani']
            self.tempaniflag[:self.fixlayers] = 0
        
        return ani

    def set_target_covariance(self, corrfix, noise_corr, rcond=None):
        # SWD noise hyper-parameters: if corr is not 0, the correlation of data
        # points assumed will be exponential.
        # RF noise hyper-parameters: if corr is not 0, but fixed, the
        # correlation between data points will be assumed gaussian (realistic).
        # if the prior for RFcorr is a range, the computation switches
        # to exponential correlated noise for RF, as gaussian noise computation
        # is too time expensive because of computation of inverse and
        # determinant each time _corr is perturbed

        for i, target in enumerate(self.targets.targets):
            target_corrfix = corrfix[i]
            target_noise_corr = noise_corr[i]

            if not target_corrfix:
                # exponential for each target
                target.get_covariance = target.valuation.get_covariance_exp
                continue

            if (target_noise_corr == 0 and np.any(np.isnan(target.obsdata.yerr))):
                # diagonal for each target, corr inrelevant for likelihood, rel error
                target.get_covariance = target.valuation.get_covariance_nocorr
                continue

            elif target_noise_corr == 0:
                # diagonal for each target, corr inrelevant for likelihood
                target.get_covariance = target.valuation.get_covariance_nocorr_scalederr
                continue

            # gauss for RF
            if target.noiseref == 'rf':
                size = target.obsdata.x.size
                target.valuation.init_covariance_gauss(
                    target_noise_corr, size, rcond=rcond)
                target.get_covariance = target.valuation.get_covariance_gauss

            # exp for noise_corr
            elif target.noiseref == 'swd':
                target.get_covariance = target.valuation.get_covariance_exp

            else:
                message = 'The noise correlation automatically defaults to the \
exponential law. Explicitly state a noise reference for your user target \
(target.noiseref) if wished differently.'
                logger.info(message)
                target.noiseref == 'swd'
                target.get_covariance = target.valuation.get_covariance_exp

    def _init_chainarrays(self, sharedmodels, sharedmisfits, sharedlikes,
                          sharednoise, sharedvpvs, sharedani):
        """from shared arrays"""
        ntargets = self.targets.ntargets
        chainidx = self.chainidx
        nchains = self.nchains

        accepted_models = int(self.iterations * np.max(self.acceptance) / 100.)
        self.nmodels = accepted_models  # 'iterations'

        msize = self.nmodels * self.maxlayers * 2
        nsize = self.nmodels * ntargets * 2
        missize = self.nmodels * (ntargets + 1)
        vpvssize = self.nmodels * self.maxlayers
        anisize = self.nmodels * self.maxlayers *3
        dtype = np.float32

        models = np.frombuffer(sharedmodels, dtype=dtype).\
            reshape((nchains, msize))
        misfits = np.frombuffer(sharedmisfits, dtype=dtype).\
            reshape((nchains, missize))
        likes = np.frombuffer(sharedlikes, dtype=dtype).\
            reshape((nchains, self.nmodels))
        noise = np.frombuffer(sharednoise, dtype=dtype).\
            reshape((nchains, nsize))
        vpvs = np.frombuffer(sharedvpvs, dtype=dtype).\
            reshape((nchains, vpvssize))
        
        if self.ani_flag:
            ani = np.frombuffer(sharedani, dtype=dtype). \
                reshape((nchains, anisize))
            self.chainani = ani[chainidx].reshape(
                self.nmodels, 3, self.maxlayers)

        self.chainmodels = models[chainidx].reshape(
            self.nmodels, self.maxlayers*2)
        self.chainmisfits = misfits[chainidx].reshape(
            self.nmodels, ntargets+1)
        self.chainlikes = likes[chainidx]
        self.chainnoise = noise[chainidx].reshape(
            self.nmodels, ntargets*2)
        self.chainvpvs = vpvs[chainidx].reshape(
            self.nmodels, self.maxlayers)
        
        self.chainiter = np.ones(self.chainlikes.size) * np.nan


# update current model (change layer number and values)

    def _model_layerbirth(self, model):
        """
        Draw a random voronoi nucleus depth from z and assign a new Vs.

        The new Vs is based on the before Vs value at the drawn z_vnoi
        position (self.propdist[2]).
        """
        n, vs_vnoi, z_vnoi = Model.split_modelparams(model)

        # new voronoi depth
        zmin, zmax = self.priors['z']
        if self.priors['fixed']: # deeper then fixed layer
            # zmin = self.fixzmax
            zmin = 10
        z_birth = self.rstate.uniform(low=zmin, high=zmax)

        ind = np.argmin((abs(z_vnoi - z_birth)))  # closest z_vnoi

        if self.priors['fixed'] and ind==self.fixlayers-1:
            ind += 1

        vs_before = vs_vnoi[ind]
        vs_birth = vs_before + self.rstate.normal(0, self.propdist[2])

        if (z_birth>self.priors['anilim'][0]) & (z_birth<self.priors['anilim'][1]):
            z_ani_flag = True
        else:
            z_ani_flag = False
        self._ani_vpvs_layerbirth(ind, z_ani_flag) # Change anisotropic parameters

        z_new = np.concatenate((z_vnoi, [z_birth]))
        vs_new = np.concatenate((vs_vnoi, [vs_birth]))

        self.dvs2 = np.square(vs_birth - vs_before)
        return np.concatenate((vs_new, z_new))

    def _model_layerdeath(self, model):
        """
        Remove a random voronoi nucleus depth from model. Delete corresponding
        Vs from model.
        """
        n, vs_vnoi, z_vnoi = Model.split_modelparams(model)
        deathlow = self.fixlayers if self.priors['fixed'] else 0 # deeper then fixed layer
        deathtop = z_vnoi.size -1 if (self.priors['fixmohoparam'] is not None) else z_vnoi.size
        ind_death = self.rstate.randint(low=deathlow, high=deathtop)
        z_before = z_vnoi[ind_death]
        vs_before = vs_vnoi[ind_death]

        z_new = np.delete(z_vnoi, ind_death)
        vs_new = np.delete(vs_vnoi, ind_death)

        ind = np.argmin((abs(z_new - z_before)))
        if (z_before>self.priors['anilim'][0]) & (z_before<self.priors['anilim'][1]):
            z_ani_flag = True
        else:
            z_ani_flag = False
        self._ani_vpvs_layerdeath(ind_death, ind, z_ani_flag) # Change anisotropic parameters

        vs_after = vs_new[ind]
        self.dvs2 = np.square(vs_after - vs_before)
        return np.concatenate((vs_new, z_new))

    def _model_vschange(self, model):
        """Randomly chose a layer to change Vs with Gauss distribution."""
        ind_top = (model.size / 2 -1) if (self.priors['fixmohoparam'] is not None) else model.size / 2
        ind_low = self.fixlayers if self.priors['fixed'] else 0
        ind = self.rstate.randint(0, ind_top)
        # ind = self.rstate.randint(0, model.size / 2)
        vs_mod = self.rstate.normal(0, self.propdist[0])
        model[ind] = model[ind] + vs_mod
        return model

    def _model_zvnoi_move(self, model):
        """Randomly choose a layer to change z_vnoi with Gaussian distribution, ensuring:
        1. All middle layers (after fixedlen) are deeper than the deepest of fixed layers
        2. The last layer is deeper than the second last layer
        No ordering constraints within fixed layers or middle layers."""
        fixlen = self.fixlayers
        while True:
            
            # Select random layer to perturb (depth parameters only)
            ind = self.rstate.randint(model.size / 2, model.size)
            original_value = model[ind]

            # Apply perturbation
            model[ind] += self.rstate.normal(0, self.propdist[1])

            depths = model[int(model.size/2):]
            n_layers = len(depths)
            valid = True

            # 1. Check last layer condition
            if self.priors['fixmohoparam'] is not None and n_layers >= 2:
                if depths[-1] <= depths[-2]:
                    valid = False

            # 2. Check middle layers vs fixed layers (if fixed layers exist)
            if self.priors['fixeddep'] is not None and fixlen < n_layers:
                max_fixed_depth = max(depths[:fixlen]) if fixlen > 0 else 0
                # Check all middle layers (from fixedlen to end-1)
                for i in range(0, fixlen):
                    if depths[i] > 10:
                        valid = False
                        break
                for i in range(fixlen, n_layers - 1):
                    if depths[i] <= max_fixed_depth:
                        valid = False
                        break
                    
                # Special case: if we only have fixedlen+1 layers,
                # the "middle" is just the last layer which has its own check
                if n_layers == fixlen + 1:
                    pass  # already checked by last layer condition
                
            if valid:
                return model
            else:
                model[ind] = original_value  # revert if invalid
                print('Deny zvnoi move')

    def _get_modelproposal(self, modify):
        model = copy.copy(self.currentmodel)

        if modify == 'vsmod':
            propmodel = self._model_vschange(model)
        elif modify == 'zvmod':
            propmodel = self._model_zvnoi_move(model)
        elif modify == 'birth':
            propmodel = self._model_layerbirth(model)
        elif modify == 'death':
            propmodel = self._model_layerdeath(model)

        return self._sort_modelproposal(propmodel, modify)

    def _sort_modelproposal(self, model, modify):
        """
        Return the sorted proposal model.

        This method is necessary, if the z_vnoi from the new proposal model
        are not ordered, i.e. if one z_vnoi value is added or strongly modified.
        """
        n, vs, z_vnoi = Model.split_modelparams(model)
        if np.all(np.diff(z_vnoi) > 0):   # monotone increasing
            return model
        else:
            ind = np.argsort(z_vnoi)
            model_sort = np.concatenate((vs[ind], z_vnoi[ind]))
            self._sort_ani_vpvs(ind, modify)
        return model_sort

    def _validmodel(self, model, **kwparam):
        """
        Check model before the forward modeling.

        - The model must contain all values > 0.
        - The layer thicknesses must be at least thickmin km.
        - if lvz: low velocity zones are allowed with the deeper layer velocity
           no smaller than (1-perc) * velocity of layer above.
        - ... and some other constraints. E.g. vs boundaries (prior) given.
        """
        modify = kwparam.pop('modify', False)
        if modify in ['birth', 'death']:
            vpvs = self.tempvpvs
        else:
            vpvs = self.currentvpvs
        vp, vs, h = Model.get_vp_vs_h(model, vpvs, self.mantle)

        # check whether nlayers lies within the prior
        layermin = self.priors['layers'][0]
        layermax = self.priors['layers'][1]
        layermodel = (h.size - 1)
        if not (layermodel >= layermin and layermodel <= layermax):
            logger.debug("chain%d: model- nlayers not in prior"
                         % self.chainidx)
            # print('Failure Valid Model Layer Number')
            return False

        # check model for layers with thicknesses of smaller thickmin
        if np.any(h[:-1] < self.thickmin):
            logger.debug("chain%d: thicknesses are not larger than thickmin"
                         % self.chainidx)
            # print('Failure Valid Model Layer Thickness')
            return False

        # check whether vs lies within the prior
        vsmin = self.priors['vs'][0]
        vsmax = self.priors['vs'][1]
        if np.any(vs < vsmin) or np.any(vs > vsmax):
            logger.debug("chain%d: model- vs not in prior"
                         % self.chainidx)
            # print('Failure Valid Model Vs')
            return False

        # check whether interfaces lie within prior
        zmin = self.priors['z'][0]
        zmax = self.priors['z'][1]
        z = np.cumsum(h)
        if np.any(z < zmin) or np.any(z > zmax):
            logger.debug("chain%d: model- z not in prior"
                         % self.chainidx)
            # print('Failure Valid Model Z')
            return False
        
        vmin, vmax = self.priors['vpvs']
        if self.priors['fixed'] and ('fixeddep' in self.priors) and (np.any(self.priors['fixeddep'])!=None):
            fixed_len = self.fixlayers
            if fixed_len > 0:
                if np.any(vpvs[:fixed_len] < vmin) or np.any(vpvs[:fixed_len] > 2.0):
                    # print('Failure Valid Model VpVs1')
                    return False
            if len(vpvs) > fixed_len:
                if np.any(vpvs[fixed_len:] < vmin) or np.any(vpvs[fixed_len:] > vmax):
                    # print(vpvs)
                    # print('Failure Valid Model VpVs2')
                    return False
        else:
            if np.any(vpvs < vmin) or np.any(vpvs > vmax):
                # print('Failure Valid Model VpVs3')
                return False

        if self.lowvelperc is not None:
            # check model for low velocity zones. If larger than perc, then
            # compvels must be positive
            compvels = vs[1:] - (vs[:-1] * (1 - self.lowvelperc))
            if not compvels.size == compvels[compvels > 0].size:
                logger.debug("chain%d: low velocity zone issues"
                             % self.chainidx)
                return False

        if self.highvelperc is not None:
            # check model for high velocity zones. If larger than perc, then
            # compvels must be positive.
            compvels = (vs[:-1] * (1 + self.highvelperc)) - vs[1:]
            if not compvels.size == compvels[compvels > 0].size:
                logger.debug("chain%d: high velocity zone issues"
                             % self.chainidx)
                return False

        if self.ani_flag and (not self.init_flag): # also for later proposal
            if modify in ['birth', 'death']:
                ani = self.tempani
            else:
                ani = self.currentani
            n, vs, z_vnoi = Model.split_modelparams(model)
            z_ani_flag = np.where((z_vnoi>self.priors['anilim'][0]) & (z_vnoi<self.priors['anilim'][1]), 0, 1)
            if self.priors['fixed']:
                z_ani_flag[:self.fixlayers] = 0
            if z_ani_flag[-1] == 0: # if including last layer
                z_ani_flag[-1] = 1
            self.tempaniflag = z_ani_flag 
            z_ani_ind = np.where(z_ani_flag == 1)[0]
            ani[:, z_ani_ind] = 0.0
        return True

    def _get_hyperparameter_proposal(self):
        noise = copy.copy(self.currentnoise)
        ind = self.rstate.choice(self.noiseinds)

        noise_mod = self.rstate.normal(0, self.propdist[3])
        noise[ind] = noise[ind] + noise_mod
        return noise

    def _validnoise(self, noise):
        for idx in self.noiseinds:
            if noise[idx] < self.noisepriors[idx][0] or \
                    noise[idx] > self.noisepriors[idx][1]:
                return False
        return True

    def _get_vpvs_proposal(self):
        vpvs = copy.copy(self.currentvpvs)
        ind_top = (vpvs.size -1) if (self.priors['fixmohoparam'] is not None) else vpvs.size
        ind = self.rstate.randint(0, ind_top)
        # ind = self.rstate.randint(0, vpvs.size)
        vpvs_mod = self.rstate.normal(0, self.propdist[4])
        vpvs[ind] = vpvs[ind] + vpvs_mod
        return vpvs

    def _validvpvs(self, vpvs):
        # only works if vpvs-priors is a range
        # if vpvs < self.priors['vpvs'][0] or \
        #         vpvs > self.priors['vpvs'][1]:

        vmin, vmax = self.priors['vpvs']
        if self.priors['fixed'] and ('fixeddep' in self.priors) and (np.any(self.priors['fixeddep'])!=None):
            fixed_len = self.fixlayers
            if fixed_len > 0:
                if np.any(vpvs[:fixed_len] < vmin) or np.any(vpvs[:fixed_len] > 2.0):
                    # print('Failure VpVs1', vpvs)
                    return False
            if len(vpvs) > fixed_len:
                if np.any(vpvs[fixed_len:] < vmin) or np.any(vpvs[fixed_len:] > vmax):

                    # print('Failure VpVs2', vpvs)
                    return False
        else:
            if np.any(vpvs < vmin) or np.any(vpvs > vmax):
                # print('Failure VpVs3', vpvs)
                return False
        return True

    def _get_ani_proposal(self, modify):
        if modify=='ani':
            ind_row = 0
        elif modify=='trend':
            ind_row = 1
        elif modify=='plunge':
            ind_row = 2
        # ind_row = self.rstate.randint(0, 3)
        ani = copy.copy(self.currentani)
        # ind_row = self.rstate.randint(0, 3)
        # ind_col = self.rstate.randint(0, ani[0].size)
        # choose ind_col by anisotropic flag
        ind_col_low = self.fixlayers if self.priors['fixed'] else 0
        ind_col = np.random.choice(np.where(self.tempaniflag[ind_col_low:]==0)[0])
        if ind_row == 0:
            # For anisotropy strength (anistr), use floating-point Gaussian perturbation
            ani_mod = round(self.rstate.normal(0, self.propdist[5]), 2)
        elif ind_row ==1:
            # For trend (ind_row=1) and plunge (ind_row=2), use rounded Gaussian for integer steps
            ani_mod = round(self.rstate.normal(0, self.propdist[6]), 1)
        else:
            ani_mod = round(self.rstate.normal(0, self.propdist[7]), 1)
        ani[ind_row, ind_col] = ani[ind_row, ind_col] + ani_mod
        # Ensure trend (ani[1, :]) is within 0-180° (wraps around)
        if ind_row == 1:
            ani[1, ind_col] = ani[1, ind_col] % 180
        # Ensure plunge (ani[2, :]) is within 0-90° (clipped)
        # elif ind_row == 2:
        #     ani[2, ind_col] = np.clip(ani[2, ind_col], 0, 90)

        return ani, ind_row, ind_col
    
    def _validani(self, ani, ind_row, ind_col):
        # only works if ani-priors is a range
        if ind_row == 0: # Strength
            if ani[ind_row, ind_col] < self.priors['anistr'][0] or \
                ani[ind_row, ind_col] > self.priors['anistr'][1]:
                return False
        elif ind_row == 1: # Trend
            if ani[ind_row, ind_col] < self.priors['anitre'][0] or \
                ani[ind_row, ind_col] > self.priors['anitre'][1]:
                return False
        elif ind_row == 2: # Plunge
            if ani[ind_row, ind_col] < self.priors['aniplu'][0] or \
                ani[ind_row, ind_col] > self.priors['aniplu'][1]:
                return False
        return True

    def _ani_vpvs_layerbirth(self, ind, z_ani_flag):
        if self.ani_flag:
            ani = copy.copy(self.currentani)
            if z_ani_flag:
                ani_birth = (ani[:, ind]).reshape((3, 1))
            else:
                ani_birth = np.zeros((3, 1))    
            self.tempani = np.concatenate((ani, ani_birth), axis=1)

        vpvs = copy.copy(self.currentvpvs)
        vpvs_birth = [vpvs[ind] + self.rstate.normal(0, self.propdist[2])]
        self.tempvpvs = np.concatenate((vpvs, vpvs_birth))
        
        # while True:
        #     valid_list = []
        #     if self.ani_flag:
        #         ani = copy.copy(self.currentani)
        #         if z_ani_flag:
        #             rand_nums = self.rstate.normal(0, [self.propdist[2], self.propdist[2], self.propdist[2]])
        #             ani_birth = (ani[:, ind] + np.array([rand_nums[0], 
        #                                                 int(round(rand_nums[1])), 
        #                                                 int(round(rand_nums[2]))])).reshape((3, 1))
        #         else:
        #             ani_birth = np.zeros((3, 1))
        #         self.tempani = np.concatenate((ani, ani_birth), axis=1)
        #         valid_list.append([self._validani(self.tempani, ind_row=i, ind_col=ind) for i in range(3)])

        #     vpvs = copy.copy(self.currentvpvs)
        #     vpvs_birth = vpvs[ind] + self.rstate.normal(0, self.propdist[2])
        #     self.tempvpvs = np.concatenate((vpvs, [vpvs_birth]))
        #     valid_list.append(self._validvpvs(self.tempvpvs))
        #     print('Generating birth ani vpvs', valid_list, self.tempvpvs)
        #     if all(valid_list):
        #         self.dvpvs2 = np.square(vpvs_birth - vpvs[ind])
        #         if z_ani_flag:
        #             self.dani2, self.dtr2, self.dplu2 = (float(np.square(ani_birth[0] - ani[0, ind])), 
        #                                                  float(np.square(ani_birth[1] - ani[1, ind])), 
        #                                                  float(np.square(ani_birth[2] - ani[2, ind])))
        #         else:
        #             self.dani2, self.dtr2, self.dplu2 = 0, 0, 0
        #         return None
    
    def _ani_vpvs_layerdeath(self, ind_death, ind, z_ani_flag):
        if self.ani_flag:
            ani = copy.copy(self.currentani)
            self.tempani = np.delete(ani, ind_death, axis=1)

        vpvs = copy.copy(self.currentvpvs)
        self.tempvpvs = np.delete(vpvs, ind_death)

        self.dvpvs2 = np.square(self.tempvpvs[ind] - vpvs[ind_death])
        if z_ani_flag:
            self.dani2, self.dtr2, self.dplu2 = (float(np.square(self.tempani[0, ind] - ani[0, ind_death])), 
                                                 float(np.square(self.tempani[1, ind] - ani[1, ind_death])), 
                                                 float(np.square(self.tempani[2, ind] - ani[2, ind_death])))
        else:
            self.dani2, self.dtr2, self.dplu2 = 0, 0, 0
        return

    def _sort_ani_vpvs(self, ind, modify):
        if modify in ['birth', 'death']:
            vpvs = self.tempvpvs
            if self.ani_flag:
                ani = self.tempani
                self.tempani  = ani[:, ind]
            self.tempvpvs = vpvs[ind]
        else:
            vpvs = copy.copy(self.currentvpvs)
            if self.ani_flag:
                ani = copy.copy(self.currentani)
                self.currentani  = ani[:, ind]
            self.currentvpvs = vpvs[ind]

        return
    
# accept / save current models

    def adjust_propdist(self):
        """
        Modify self.propdist to adjust acceptance rate of models to given
        percentace span: increase or decrease by five percent.
        """
        with np.errstate(invalid='ignore'):
            acceptrate = self.accepted / self.proposed * 100
        # minimum distribution width forced to be not less than 1 m/s, 1 m
        # actually only touched by vs distribution
        propdistmin = np.full(acceptrate.size, 0.001)
        # logger.info('%s' % acceptrate)
        for i, rate in enumerate(acceptrate):
            if np.isnan(rate):
                # only if not inverted for
                continue
            if rate < self.acceptance[0]:
                new = self.propdist[i] * 0.95
                if new < propdistmin[i]:
                    new = propdistmin[i]
                self.propdist[i] = new

            elif rate > self.acceptance[1]:
                self.propdist[i] = self.propdist[i] * 1.05
            else:
                pass

    def get_acceptance_probability(self, modify):
        """
        Acceptance probability will be computed dependent on the modification.

        Parametrization alteration (Vs or voronoi nuclei position)
            the acceptance probability is equal to likelihood ratio.

        Model dimension alteration (layer birth or death)
            the probability was computed after the formulation of Bodin et al.,
            2012: 'Transdimensional inversion of receiver functions and
            surface wave dispersion'.
        """
        # if modify in ['vsmod', 'zvmod', 'noise', 'vpvs', 'ani']:
        if modify in ['vsmod', 'zvmod', 'noise', 'vpvs', 'ani', 'trend', 'plunge']:
            # only velocity or thickness changes are made
            # also used for noise changes
            alpha = self.targets.proposallikelihood - self.currentlikelihood
            

        elif modify in ['birth', ]:
            theta = self.propdist[2]  # Gaussian distribution
            sqrt_2pi = np.sqrt(2 * np.pi)

            # self.dvs2 = delta vs square = np.square(v'_(k+1) - v_(i))
            A = (theta * sqrt_2pi) / self.dv
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood

            alpha = np.log(A) + B + C
            # if type(self.priors['vpvs']) != float:
            #     sigma_vpvs = self.propdist[4]
            #     alpha = alpha + np.log((self.propdist[4] * sqrt_2pi) / self.dvpvs) + self.dvpvs2 / (2. * np.square(sigma_vpvs))
            # if self.ani_flag:
                # sigma_ani = self.propdist[5]   # Gaussian sigma for anisotropy
                # sigma_tr_plu = self.propdist[6]  # Gaussian sigma for trend/plunge
                # sigma_tr_plu_sq = np.square(sigma_tr_plu)
                # if self.dani2 != 0:
                #     delta_alpha = np.log((sigma_ani * sqrt_2pi) / self.dani) + self.dani2 / (2. * np.square(sigma_ani)) +\
                #                     np.log((sigma_tr_plu * sqrt_2pi) / self.dtr) + self.dtr2 / (2. * sigma_tr_plu_sq) +\
                #                     np.log((sigma_tr_plu * sqrt_2pi) / self.dplu) + self.dplu2 / (2. * sigma_tr_plu_sq)
                #     alpha = alpha + delta_alpha
                #     print(f'ani: {delta_alpha}')
        elif modify in ['death', ]:
            theta = self.propdist[2]  # Gaussian distribution
            sqrt_2pi = np.sqrt(2 * np.pi)

            # self.dvs2 = delta vs square = np.square(v'_(j) - v_(i))
            A = self.dv / (theta * sqrt_2pi)
            B = self.dvs2 / (2. * np.square(theta))
            C = self.targets.proposallikelihood - self.currentlikelihood

            alpha = np.log(A) - B + C
            # if type(self.priors['vpvs']) != float:
            #     sigma_vpvs = self.propdist[4]
            #     alpha = alpha + np.log((self.propdist[4] * sqrt_2pi) / self.dvpvs) - self.dvpvs2 / (2. * np.square(sigma_vpvs))
            # if self.ani_flag:
            #     sigma_ani = self.propdist[5]   # Gaussian sigma for anisotropy
            #     sigma_tr_plu = self.propdist[6]  # Gaussian sigma for trend/plunge
            #     sigma_tr_plu_sq = np.square(sigma_tr_plu)
                # if self.dani2 != 0:
                #     delta_alpha = np.log(self.dani / (sigma_ani * sqrt_2pi)) - self.dani2 / (2. * np.square(sigma_ani)) +\
                #                     np.log(self.dtr / (sigma_tr_plu * sqrt_2pi)) - self.dtr2 / (2. * sigma_tr_plu_sq) +\
                #                     np.log(self.dplu / (sigma_tr_plu * sqrt_2pi)) - self.dplu2 / (2. * sigma_tr_plu_sq)
                #     alpha = alpha + delta_alpha
                #     print(f'ani: {delta_alpha}')
        return alpha

    def accept_as_currentmodel(self, model, noise, vpvs, **kwargs):
        """Assign currentmodel and currentvalues to self."""
        ani = kwargs.get('ani', np.zeros((3, vpvs.size)))
        flag = kwargs.get('flag', np.ones(vpvs.size))
        self.currentmisfits = self.targets.proposalmisfits
        self.currentlikelihood = self.targets.proposallikelihood
        self.currentmodel = model
        self.currentnoise = noise
        self.currentvpvs = vpvs
        self.currentani = ani
        self.currentaniflag = flag
        self.lastmoditer = self.iiter
        if self.priors['fixmohoparam'] is not None:
            self.priors['z'][1] = model[-1]

    def append_currentmodel(self):
        """Append currentmodel to chainmodels and values."""
        self.chainmodels[self.n, :self.currentmodel.size] = self.currentmodel
        self.chainmisfits[self.n, :] = self.currentmisfits
        self.chainlikes[self.n] = self.currentlikelihood
        self.chainnoise[self.n, :] = self.currentnoise
        self.chainvpvs[self.n, :int(self.currentmodel.size/2)] = self.currentvpvs
        if self.ani_flag:
            self.chainani[self.n, :, :int(self.currentmodel.size/2)] = self.currentani
        self.chainiter[self.n] = self.iiter
        self.n += 1

# run optimization

    def iterate(self):
        ani_ind_low = self.fixlayers if self.priors['fixed'] else 0 # Judge whether ani exist below fixed
        layermodel = int(self.currentmodel.size / 2)
        # if (self.iiter > (-self.iter_phase1 + (self.iterations * 0.01))) & np.any(self.currentaniflag[ani_ind_low:] == 0):
        #     modify = self.rstate.choice(self.modifications + self.animods)

        # elif (self.iiter < (-self.iter_phase1 + (self.iterations * 0.01))) & (np.any(self.currentaniflag[ani_ind_low:] == 0)): 
        #     # only allow vs and z modifications the first 1 % of iterations
        #     modify = self.rstate.choice(['vsmod', 'zvmod'] + self.noisemods + self.vpvsmods + self.animods)
            
        # elif self.iiter < (-self.iter_phase1 + (self.iterations * 0.01)):
        #     # only allow vs and z modifications the first 1 % of iterations
        #     modify = self.rstate.choice(['vsmod', 'zvmod'] + self.noisemods + self.vpvsmods)
            
        # else:
        #     modify = self.rstate.choice(self.modifications)

        if self.iiter <= int(-self.iter_phase1 + (self.iterations * 0.01)):
            # 总迭代的前1%：仅允许 vsmod/zvmod + noise/vpvs，不允许 birth/death/animods
            modify_options = ['vsmod', 'zvmod'] + self.noisemods + self.vpvsmods
            modify = self.rstate.choice(modify_options)
        elif self.iiter <= int(-self.iter_phase1 + (self.iterations * 0.3)):
            # iter_phase1 的前30%：允许 vsmod/zvmod + noise/vpvs + birth/death，但不允许 animods
            modify_options = copy.copy(self.modifications)
            if layermodel==self.maxlayers:
                modify_options.remove('birth')
            modify = self.rstate.choice(modify_options)
        else:
            # 其余情况：允许所有 modifications + animods（如果 currentaniflag 允许）
            if np.any(self.currentaniflag[ani_ind_low:] == 0):
                modify_options = self.modifications + self.animods
            else:
                modify_options = copy.copy(self.modifications)
            if layermodel==self.maxlayers:
                modify_options.remove('birth')
            modify = self.rstate.choice(modify_options)

        self.tempaniflag = self.currentaniflag
        proposalani = self.currentani

        if modify in self.modelmods:
            proposalmodel = self._get_modelproposal(modify)
            if modify in ['birth', 'death']:
                proposalvpvs = self.tempvpvs
                if self.ani_flag:
                    proposalani = self.tempani
            else:
                proposalvpvs = self.currentvpvs
                proposalani = self.currentani
            proposalnoise = self.currentnoise
            if not self._validmodel(proposalmodel, modify=modify):
                proposalmodel = None

        elif modify in self.noisemods:
            proposalmodel = self.currentmodel
            proposalnoise = self._get_hyperparameter_proposal()
            proposalvpvs = self.currentvpvs
            proposalani = self.currentani
            if not self._validnoise(proposalnoise):
                proposalmodel = None

        elif modify == 'vpvs':
            proposalmodel = self.currentmodel
            proposalnoise = self.currentnoise
            proposalvpvs = self._get_vpvs_proposal()
            proposalani = self.currentani
            if not self._validvpvs(proposalvpvs):
                proposalmodel = None

        elif modify in self.animods:
            proposalmodel = self.currentmodel
            proposalnoise = self.currentnoise
            proposalvpvs = self.currentvpvs
            proposalani, ind_row, ind_col = self._get_ani_proposal(modify=modify)
            if not self._validani(proposalani, ind_row, ind_col):
                proposalmodel = None

        if proposalmodel is None:
            # If not a valid proposal model and noise params are found,
            # leave self.iterate and try with another modification
            # should not occur often.
            logger.debug('Not able to find a proposal for %s' % modify)
            self.iiter += 1
            return
        # compute synthetic data and likelihood, misfit
        
        # stage = False if self.iiter <= int(-self.iter_phase1 + (self.iterations * 0.3)) else True # frist 30% of burn-in only 16 trace 
        stage = False
        if self.priors['fixvpvs']:
            proposalvpvs = self.fix_vpvs(proposalvpvs, proposalmodel[int(len(proposalmodel)/2):])
        vp, vs, h = Model.get_vp_vs_h(proposalmodel, proposalvpvs, self.mantle)
        self.targets.evaluate(h=h, vp=vp, vs=vs, noise=proposalnoise, ani=proposalani, flag=self.tempaniflag, stage=stage)
        # if modify=='ani':
        #     if ind_row == 1:
        #         modify = 'trend'
        #     elif ind_row == 2:
        #         modify = 'plunge'
        paridx = PAR_MAP[modify]
        self.proposed[paridx] += 1

        # Replace self.currentmodel with proposalmodel with acceptance
        # probability alpha. Accept candidate sample (proposalmodel)
        # with probability alpha, or reject it with probability (1 - alpha).
        # these are log values ! alpha is log.
        u = np.log(self.rstate.uniform(0, 1))
        alpha = self.get_acceptance_probability(modify)
        # if modify in ['birth', 'death']:
        #     print(f"u: {u}; alpha: {alpha}; modify: {modify}")
        # #### _____________________________________________________________
        if u < alpha:
            # always the case if self.jointlike > self.bestlike (alpha>1)
            self.accept_as_currentmodel(proposalmodel, proposalnoise, proposalvpvs, ani=proposalani, flag=self.tempaniflag)
            self.append_currentmodel()
            self.accepted[paridx] += 1
        # else:
        #     logger.info(f'proposallikelihood:{self.targets.proposallikelihood}, currentlikelihood:{self.currentlikelihood}')
        #     logger.info('Not able to find a proposal for %s' % modify)
        # print inversion status information
        if self.iiter % 2000 == 0:
            runtime = time.time() - self.tnull
            current_iterations = self.iiter + self.iter_phase1
            if current_iterations > 0:
                acceptrate = float(self.n) / current_iterations * 100.
                logger.info('%6d %5d + hs %8.3f\t%9d |%6.1f s  | %.1f ' % (
                    self.lastmoditer, self.currentmodel.size/2 - 1,
                    self.currentmisfits[-1], self.currentlikelihood,
                    runtime, acceptrate) + r'%')

            self.tnull = time.time()

        # stabilize model acceptance rate
        if self.iiter % 3000 == 0:
            if self.priors['fixvpvs']:
                self.proposed[4] = 1
            # logger.info('Current Proposal %s' % (self.proposed))
            if np.all(self.proposed) != 0:
                self.adjust_propdist()
        self.iiter += 1

    def run_chain(self):
        t0 = time.time()
        self.tnull = time.time()
        self.iiter = -self.iter_phase1

        self.modelmods = ['vsmod', 'zvmod', 'birth', 'death']
        self.noisemods = [] if len(self.noiseinds) == 0 else ['noise']
        self.vpvsmods = [] if (type(self.priors['vpvs']) == float or self.priors['fixvpvs']) else ['vpvs']
        # self.animods = ['ani'] if self.ani_flag else []
        self.animods = ['ani', 'trend', 'plunge'] if self.ani_flag else []

        self.modifications = self.modelmods + self.noisemods + self.vpvsmods

        self.accepted = np.zeros(len(self.propdist))
        self.proposed = np.zeros(len(self.propdist))
        while self.iiter < self.iter_phase2:
            # t1 = time.time()
            self.iterate()
            # print(f'Iteration cost {time.time()-t1}')
        runtime = (time.time() - t0)

        # update chain values (eliminate nan rows)
        self.chainmodels = self.chainmodels[:self.n, :]
        self.chainmisfits = self.chainmisfits[:self.n, :]
        self.chainlikes = self.chainlikes[:self.n]
        self.chainnoise = self.chainnoise[:self.n, :]
        self.chainvpvs = self.chainvpvs[:self.n, :]
        if self.ani_flag:
            self.chainani = self.chainani[:self.n, :, :]
        self.chainiter = self.chainiter[:self.n]

        # only consider models after burnin phase
        p1ind = np.where(self.chainiter < 0)[0]
        p2ind = np.where(self.chainiter >= 0)[0]
        if p1ind.size != 0:
            wmodels, wlikes, wmisfits, wnoise, wvpvs, wani = self.get_weightedvalues(
                pind=p1ind, finaliter=0)
            self.p1models = wmodels  # p1 = phase one
            self.p1misfits = wmisfits
            self.p1likes = wlikes
            self.p1noise = wnoise
            self.p1vpvs = wvpvs
            self.p1ani = wani

        if p2ind.size != 0:
            wmodels, wlikes, wmisfits, wnoise, wvpvs, wani = self.get_weightedvalues(
                pind=p2ind, finaliter=self.iiter)
            self.p2models = wmodels  # p2 = phase two
            self.p2misfits = wmisfits
            self.p2likes = wlikes
            self.p2noise = wnoise
            self.p2vpvs = wvpvs
            self.p2ani = wani

        accmodels = float(self.p2likes.size)  # accepted models in p2 phase
        maxmodels = float(self.initparams['maxmodels'])  # for saving
        self.thinning = int(np.ceil(accmodels / maxmodels))
        self.save_finalmodels()

        logger.debug('time for inversion: %.2f s' % runtime)

    def get_weightedvalues(self, pind, finaliter):
        """
        Models will get repeated (weighted).

        Each iteration, if there was no model proposal accepted, the current
        model gets repeated once more. This weight is based on self.chainiter,
        which documents the iteration of the last accepted model."""
        pmodels = self.chainmodels[pind]  # p = phase (1 or 2)
        pmisfits = self.chainmisfits[pind]
        plikes = self.chainlikes[pind]
        pnoise = self.chainnoise[pind]
        pvpvs = self.chainvpvs[pind]
        if self.ani_flag:
            pani = self.chainani[pind]
        else:
            pani = None
        pweights = np.diff(np.concatenate((self.chainiter[pind], [finaliter])))

        wmodels, wlikes, wmisfits, wnoise, wvpvs, wani = ModelMatrix.get_weightedvalues(
            pweights, models=pmodels, likes=plikes, misfits=pmisfits,
            noiseparams=pnoise, vpvss=pvpvs, anis=pani)
        return wmodels, wlikes, wmisfits, wnoise, wvpvs, wani

    def save_finalmodels(self):
        """Save chainmodels as pkl file"""
        savepath = op.join(self.initparams['savepath'], 'data')
        names = ['models', 'likes', 'misfits', 'noise', 'vpvs', 'ani']

        # phase 1 -- burnin
        try:
            for i, data in enumerate([self.p1models, self.p1likes,
                                     self.p1misfits, self.p1noise,
                                     self.p1vpvs, self.p1ani]):
                outfile = op.join(savepath, 'c%.3d_p1%s' % (self.chainidx, names[i]))
                if data is None or (isinstance(data, np.ndarray) and data.size == 0):
                    continue
                np.save(outfile, data[::self.thinning])
        except:
            logger.info('No burnin models accepted.')

        # phase 2 -- main / posterior phase
        try:
            for i, data in enumerate([self.p2models, self.p2likes,
                                     self.p2misfits, self.p2noise,
                                     self.p2vpvs, self.p2ani]):
                outfile = op.join(savepath, 'c%.3d_p2%s' % (self.chainidx, names[i]))
                if data is None or (isinstance(data, np.ndarray) and data.size == 0):
                    continue
                np.save(outfile, data[::self.thinning])

            logger.info('> Saving %d models (main phase).' % len(data[::self.thinning]))
        except:
            logger.info('No main phase models accepted.')
