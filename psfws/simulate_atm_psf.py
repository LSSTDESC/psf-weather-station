import os
import psfws
import numpy as np
import galsim
import pickle
import logging
from galsim.config import InputLoader, RegisterInputType, RegisterObjectType
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
import astropy.units as u

from scipy.optimize import bisect

# Heavily based on: https://github.com/LSSTDESC/imSim/blob/main/imsim/atmPSF.py

# For now:
# - screen size set by user
# - no multithreading
# - seeing calibrated to user provided value


class AtmosphericPSF():
    """Class representing an Atmospheric PSF.

    A random realization of the atmosphere will be produced when this class is
    instantiated. Realistic weather parameters from psf-weather-station."""
    def __init__(self, alt, az, band, rng, boresight, rawSeeing, 
                 exptime=30.0, t0=0.0, nlayers=6, screen_size=800, exponent=-0.3,
                 kcrit=0.2, screen_scale=0.1, save_file=None, logger=None,
<<<<<<< HEAD
                 field_x=None, field_y=None, angle_random=False):
=======
                 field_x=None, field_y=None):
>>>>>>> aa4c0c62f5d5f05c069521140bcdbbf3631cb040
        self.boresight = boresight
        self.alt = alt
        self.az = az
        self.targetFWHM = rawSeeing
        self.logger = galsim.config.LoggerWrapper(logger)
        self.exponent = exponent
<<<<<<< HEAD
        self.angle_random = angle_random
=======
>>>>>>> aa4c0c62f5d5f05c069521140bcdbbf3631cb040

        self.wlen_eff = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)[band]
        # wlen_eff is from Table 2 of LSE-40 (y=y2)

        self.rng = rng
        self.t0 = t0
        self.exptime = exptime
        self.screen_size = screen_size
        self.screen_scale = screen_scale
        self.nlayers = nlayers
        
        if save_file and os.path.isfile(save_file):
            self.logger.warning(f'Reading atmospheric PSF from {save_file}')
            self.load_psf(save_file)
        else:
            self.logger.warning('Building atmospheric PSF')
            self._build_atm(kcrit)
            if save_file:
                self.logger.warning(f'Saving atmospheric PSF to {save_file}')
                self.save_psf(save_file)

    def save_psf(self, save_file):
        """
        Save the psf as a pickle file.
        """
        with open(save_file, 'wb') as fd:
            with galsim.utilities.pickle_shared():
                pickle.dump((self.atm, self.aper), fd)

    def load_psf(self, save_file):
        """
        Load a psf from a pickle file.
        """
        with open(save_file, 'rb') as fd:
            self.atm, self.aper = pickle.load(fd)

    def _build_atm(self, kcrit):
        atm_kwargs = self._getAtmKwargs()
        self.atm = galsim.Atmosphere(**atm_kwargs)
        self.aper = galsim.Aperture(diam=8.36, obscuration=0.61,
                                    lam=self.wlen_eff, screen_list=self.atm)

        # Instantiate screens now instead of delaying until after multiprocessing
        # has started.
        r0 = atm_kwargs['r0_500'] * (self.wlen_eff/500.0)**(6./5)
        kmax = kcrit / r0

        self.logger.info("Instantiating atmospheric screens")

        self.atm.instantiate(kmax=kmax, check='phot')

        self.logger.info("Finished building atmosphere")
        self.logger.debug("GSScreenShare keys = %s",list(galsim.phase_screens._GSScreenShare.keys()))
        self.logger.debug("id(self) = %s",id(self))

    def _vkSeeing(self, r0_500, wavelength, L0):
        # von Karman profile FWHM from fitting formula in eqn 19 of
        # Tokovinin 2002, PASP, v114, p1156
        # https://dx.doi.org/10.1086/342683
        kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
        r0 = r0_500 * (wavelength/500)**(6./5)
        arg = 1. - 2.183*(r0/L0)**0.356
        factor = np.sqrt(arg) if arg > 0.0 else 0.0
        return kolm_seeing*factor

    def _seeingResid(self, r0_500, wavelength, L0, targetSeeing):
        return self._vkSeeing(r0_500, wavelength, L0) - targetSeeing

    def _r0_500(self, wavelength, L0, targetSeeing):
        """Returns r0_500 to use to get target seeing."""
        r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**(6./5))
        r0_500_min = 0.01
        return bisect(
            self._seeingResid,
            r0_500_min,
            r0_500_max,
            args=(wavelength, L0, targetSeeing)
        )
    
    def _getAtmKwargs(self):
        """Get all atmospheric setup parameters."""
        # psf-weather-station params
        speeds, directions, altitudes, weights = self._get_psfws_params()

<<<<<<< HEAD
        if self.angle_random:
            # randomize wind directions; useful for testing purposes
            ud = galsim.UniformDeviate(self.rng)
            directions = [ud()*360*galsim.degrees for _ in range(len(directions))]

=======
>>>>>>> aa4c0c62f5d5f05c069521140bcdbbf3631cb040
        # Draw L0 from truncated log normal, broadcast to list of layers
        gd = galsim.GaussianDeviate(self.rng)
        L0 = 0
        while L0 < 10.0 or L0 > 100:
            L0 = np.exp(gd() * 0.6 + np.log(25.0))    
        L0 = [L0] * len(speeds)

        # associated r0 at 500nm for these turbulence weights
        r0_500 = self._r0_500(self.wlen_eff, L0[0], self.targetFWHM)
#         r0_500 = (2.914 * (500e-9)**(-2) * np.sum(weights))**(-3./5) / 5 # fudge factor of 1/5

        atmKwargs = dict(
            r0_500=r0_500,
            L0=L0, 
            speed=speeds,
            direction=directions,
            altitude=altitudes,
            r0_weights=weights,
            screen_size=self.screen_size, 
            screen_scale=self.screen_scale,
            rng=self.rng
        )

        if self.logger:
            self.logger.debug("airmass = {}".format(1 / np.cos(np.pi/2-self.alt.rad)))
            self.logger.debug("wlen_eff = {}".format(self.wlen_eff))
            self.logger.debug("r0_500 = {}".format(r0_500))
            self.logger.debug("L0 = {}".format(L0))
            self.logger.debug("speeds = {}".format(speeds))
            self.logger.debug("directions = {}".format(directions))
            self.logger.debug("altitudes = {}".format(altitudes))
            self.logger.debug("weights = {}".format(weights))

        return atmKwargs

    def _set_screen_size(self, speeds):
        vmax = np.max(speeds)
        if vmax > 35:
            screen_size = 1050
        else:
            screen_size = vmax * 30
        return screen_size

    def _get_psfws_params(self):
        """Use psf-weather-station to fetch simulation setup parameters."""
        ws = psfws.ParameterGenerator(seed=self.rng.raw())
        pt = ws.draw_datapoint()

        params = ws.get_parameters(pt, nl=self.nlayers, skycoord=True, 
                                   alt=self.alt, az=self.az, location='com')
        
        # place layers 200m above ground
        altitudes = [p - ws.h0 + 0.2 for p in params['h']]
        directions = [i*galsim.degrees for i in params['phi']]
        
        return params['speed'], directions, altitudes, params['j']

    def getPSF(self, field_pos, gsparams=None):
        """
        Return a PSF to be convolved with sources.

        @param [in] field position of the object relative to the boresight direction.
        """
        theta = (field_pos.x*galsim.arcsec, field_pos.y*galsim.arcsec)

        psf = galsim.ChromaticAtmosphere(
                  self.atm.makePSF(self.wlen_eff, 
                                   aper=self.aper, 
                                   theta=theta, 
                                   t0=self.t0,
                                   exptime=self.exptime, 
                                   gsparams=gsparams),
                  alpha=self.exponent,
                  base_wavelength=self.wlen_eff,
                  zenith_angle=0*galsim.degrees
              )
        return psf


class AtmLoader(InputLoader):
    """Custom AtmosphericPSF loader that only loads the atmosphere once per exposure.

    Note: For now, this just loads the atmosphere once for an entire imsim run.
          If we ever decide we want to have a single config processing run handle multiple
          exposures (rather than just multiple CCDs for a single exposure), we'll need to
          reconsider this implementation.
    """
    def __init__(self):
        # Override some defaults in the base init.
        super().__init__(init_func=AtmosphericPSF,
                         takes_logger=True, use_proxy=False,
                         worker_init=galsim.phase_screens.initWorker,
                         worker_initargs=galsim.phase_screens.initWorkerArgs)

    def getKwargs(self, config, base, logger):
        logger.debug("Get kwargs for AtmosphericPSF")

        req_params = { 
                       'band' : str,
                       'boresight' : galsim.CelestialCoord,
                       'rawSeeing' : float,
                       'alt' : galsim.Angle,
                       'az' : galsim.Angle
                     }
        opt_params = { 
                       't0' : float,
                       'exptime' : float,
                       'kcrit' : float,
                       'screen_size' : float,
                       'screen_scale' : float,
                       'save_file' : str,
                       'nlayers' : int,
<<<<<<< HEAD
                       'exponent' : float,
                       'angle_random' : bool,
=======
                       'exponent' : float
>>>>>>> aa4c0c62f5d5f05c069521140bcdbbf3631cb040
                     }

        # Temporary fix until GalSim 2.5 to make sure atm_psf can be built once and shared,
        # even if the opsim_data that it often needs is later in the list of inputs.
        try:
            kwargs, _ = galsim.config.GetAllParams(config, base, req=req_params, opt=opt_params)
        except galsim.GalSimError as e:
            if str(e).startswith("No input opsim_data"):
                galsim.config.LoadInputObj(base, 'opsim_data', 0, True, logger)
                kwargs, _ = galsim.config.GetAllParams(config, base, req=req_params, opt=opt_params)
            else:
                raise

        logger.debug("kwargs = %s",kwargs)

        # We want this to be set up right at the beginning of the run, before the config
        # stuff has even set up the RNG yet.  So make an rng ourselves based on the
        # random seed in image.random_seed.

        seed = base['image'].get('random_seed', None)
        if seed is None:
            raise RuntimeError("AtmLoader requires a seed in config['image']['random_seed']")
        if isinstance(seed, list):
            seed = seed[0]  # If random_seed is a list, just use the first one.
        # Parse the value in case it is an eval or something.
        seed = galsim.config.ParseValue({'seed': seed}, 'seed', base, int)[0]
        # Somewhat gratuitously add an aribtary value to this to avoid any correlations with
        # other uses of this random seed elsewhere in the config processing.
        seed += 271828
        rng = galsim.BaseDeviate(seed)
        kwargs['rng'] = rng
        logger.debug("seed for atm = %s",seed)

        # Include the logger
        kwargs['logger'] = logger

        # safe=True means this will be used for the whole run.
        safe = True

        return kwargs, safe


def BuildPsfwsAtmosphericPSF(config, base, ignore, gsparams, logger):
    """Build an AtmosphericPSF from the information in the config file.

    Built to interface with ImSim using OpSim inputs.
    """
    atm = galsim.config.GetInputObj('atm_psf', config, base, 'AtmosphericPSF')
    image_pos = base['image_pos']
    boresight = atm.boresight
    field_pos = base['wcs'].posToWorld(image_pos, project_center=boresight)
    if gsparams: gsparams = galsim.GSParams(**gsparams)
    else: gsparams = None

    #logger.debug("Making PSF for pos %s",image_pos)
    #logger.debug("GSScreenShare keys = %s",list(galsim.phase_screens._GSScreenShare.keys()))
    #logger.debug("type(atm) = %s",str(type(atm)))
    #logger.debug("id(atm) = %s",id(atm))
    psf = atm.getPSF(field_pos, gsparams)
    return psf, False

def BuildAtmosphericPSF(config, base, ignore, gsparams, logger):
    """Build an AtmosphericPSF from the information in the config file.

    Version without input loading module for simpler sims.
    """
    config['rng'] = base['rng']
    config['boresight'] = None
<<<<<<< HEAD
    rq = {'alt': galsim.Angle,
          'az' : galsim.Angle,
          'band' : str,
          'rng': galsim.BaseDeviate,
         }
    op = {'field_x' : float,
          'field_y' : float,
          't0' : float,
          'exptime' : float,
          'rawSeeing': float,
          'kcrit' : float,
          'screen_size' : float,
          'screen_scale' : float,
          'exponent' : float,
          'boresight' : galsim.CelestialCoord,
          'angle_random' : bool,
         }
=======
    rq = {'alt': galsim.Angle, 'az' : galsim.Angle , 'band' : str, 'rng': galsim.BaseDeviate}
    op = {'field_x' : float, 'field_y' : float, 't0' : float, 'exptime' : float, 'rawSeeing': float,
          'kcrit' : float, 'screen_size' : float, 'screen_scale' : float, 'exponent' : float,
          'boresight' : galsim.CelestialCoord}
>>>>>>> aa4c0c62f5d5f05c069521140bcdbbf3631cb040
    kwargs, _ = galsim.config.GetAllParams(config, base, req=rq, opt=op)
    if gsparams: gsparams = galsim.GSParams(**gsparams)
    else: gsparams = None

    atm = AtmosphericPSF(**kwargs)
    psf = atm.getPSF(galsim.PositionD(config['field_x'], config['field_y']), gsparams)
    return psf, False

RegisterInputType('atm_psf', AtmLoader())
RegisterObjectType('PsfwsAtmosphericPSF', BuildPsfwsAtmosphericPSF, input_type='atm_psf')

RegisterObjectType('AtmosphericPSF', BuildAtmosphericPSF)
