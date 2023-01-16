"""Class to generate realistic input parameters for atmospheric PSF sims."""

import numpy as np
import pickle
import pandas as pd
import pathlib
from . import utils

data_dir = pathlib.Path(utils.get_data_path())

class ParameterGenerator():
    """Class to generate realistic input parameters for atmospheric PSF sims.

    This class uses as main input global circulation model weather forecasting
    outputs, from either the NOAA Global Forecasting System (GFS) analysis or
    the European Center for Midrange Weather Forecasting (ECMWF) reanalysis 
    dataset ERA5.
    Optionally, local wind measurements from the site of interest may be used
    to improve the accuracy of the outputs. The package contains these data
    for Cerro Pachon, and all defaults are set up to match this location.
    Use of the code to generate parameters for Cerro Pachon (and Cerro 
    Telolo, nearby) is straightforward, but for use at other observatories,
    users must supply input data: instructions for downloading and formatting 
    forecasting data/telemetry are in the README.

    Parameters:
        seed:           Random seed to initialize rng [default: None]
        h_tel:          Altitude, in km, of the observatory. [default: 2.715]
        rho_jv:         Desired correlation coefficient between the ground wind
                        speed and the ground turbulence integral. If a nonzero
                        float value is specified, the joint PDF of wind values
                        and ground turbulence is generated and the turbulence
                        samples are stored in the ``data_gl`` attribute as the
                        ``j_gl`` column. [default: 0]
        turbulence:     Dictionary of lognormal PDF parameters (``s`` and 
                        ``scale``) describing the statistics of turbulence in 
                        the ground layer (``gl``) and free atmosphere (``fa``). 
                        [Default: {'gl': {'s': 0.62, 'scale': 2.34},
                                   'fa': {'s': 0.84, 'scale': 1.51}]
                        Turbulence PDF parameters from 
                        https://doi.org/10.1111/j.1365-2966.2005.09813.x]
        forecast_file:  Path to weather forecast data. [default:
                        'data/gfs_-30.0_289.5_20190501-20191101.pkl']
        telemetry_file: Path to telemetry data. If None, forecast data
                        will be used for ground level information. [default:
                        'data/tel_dict_CP_20190501-20191101.pkl']

    Attributes:
        h0:             Altitude of observatory. Measured in km with respect to 
                        sea level.
        h:              Ndarray of forecasting data altitudes above ``h0``. 
                        Measured in km with respect to sea level. 
        p:              Ndarray of forecast pressures (mbar).
        data_fa:        Pandas dataframe of forecasting data, with columns 
                        [``u``, ``v``, ``speed``, ``phi``, ``t``]. Each entry is
                        a ndarray of the same length as ``h`` and the index of
                        the dataframe are DateTime objects, i.e. one df entry
                        corresponds to simultaneous outputs at all altitudes.
                        Velocity (``u``,``v`` are the east/northward components)
                        and speed are given in m/s, temperature in Kelvin, and 
                        ``phi``, using the meteorological convention of the 
                        direction of wind origin, in degrees east of north.
        data_gl:        Pandas dataframe of ground level data, either telemetry
                        (if a data file was given) temporally matched to 
                        ``data_fa``, or forecast data interpolated to ground 
                        altitude. Index and columns are the same as ``data_fa``,
                        unless ``rho_jv`` was specified, in which case a column
                        ``j_gl`` is added with ground turbulence samples.
        rho_jv:         Correlation coefficient between the ground level wind 
                        speed and ground turbulence integral.
        fa_start:       Integer index of ``h`` corresponding to the start of the
                        free atmosphere (800m above ground). For example, to 
                        access ``u`` values in the FA for some Index ``pt``: 
                        ``data_fa.at[pt, 'u'][fa_start:]``
        j_pdf:          Dictionary containing parameters for lognormal PDFs of 
                        empirical turbulence integral distributions for the 
                        ground layer (``gl``) and free atmosphere (``fa``).
        N:              Number of matched datasets.

    Notes:
        Code is written to output quantities formatted to match desired inputs
        for GalSim atompsheric PSF simulations.

    """

    def __init__(self, seed=None, h_tel=2.715, rho_jv=0, turbulence=None,
                 forecast_file='ecmwf_-30.25_-70.75_20190501_20191031.p',
                 telemetry_file='tel_dict_CP_20190501-20191101.pkl'):
        # set up the paths to data files, and check they exist.
        self._paths = \
            {'forecast_data': pathlib.Path.joinpath(data_dir, forecast_file),
             'p_and_h': pathlib.Path.joinpath(data_dir, 'p_and_h.p')}

        if telemetry_file is not None:
            self._paths['telemetry'] = pathlib.Path.joinpath(data_dir, 
                                                             telemetry_file)
            use_telemetry = True
        else:
            use_telemetry = False
             
        for file_path in self._paths.values():
            if not file_path.is_file():
                print(f'code running from: {data_dir}')
                raise FileNotFoundError(f'file {file_path} not found!')

        if turbulence is None:
            turbulence = {'gl': {'s': 0.62, 'scale': 2.34},
                          'fa': {'s': 0.84, 'scale': 1.51}}

        # set ground altitude, turbulence pdf, and j/wind correlation
        self.h0 = h_tel 
        self.h_atm_end = 25  # km
        self.gl_height = 0.8  # km

        self.j_pdf = {k: utils.lognorm(v['s'], v['scale']) for k, v in turbulence.items()}
        self.rho_jv = rho_jv

        self._rng = np.random.default_rng(seed)

        # load and match forecast/telemetry data
        self._load_data(use_telemetry)

        # if using correlation between wind speed and ground turbulence,
        # draw values in advance and perform correlation of marginals
        if self.rho_jv != 0:
            # draw JGL values
            j_gl = self.j_pdf['gl'].rvs(size=self.N, random_state=self._rng)
            # correlate and store modified dataframe
            self.data_gl = utils.correlate_marginals(self.data_gl,
                                                     j_gl,
                                                     self.rho_jv,
                                                     self._rng)

    def _load_data(self, use_telemetry=True):
        """Load data from forecast, telemetry files, match, and store."""
        forecast = pickle.load(open(self._paths['forecast_data'], 'rb'))
        forecast = utils.process_forecast(forecast)

        # load heights and pressures
        p_and_h = pickle.load(open(self._paths['p_and_h'], 'rb'))
        src = 'ecmwf' if len(forecast['u'].iat[0]) > 50 else 'noaa'
        
        # reverse (input was in reverse height order), convert from m to km
        h = p_and_h[src]['h'][::-1] / 1000
        # find lower gl cutoff (h is sorted due to previous line)
        where_h0 = np.searchsorted(h, self.h0)
        # free atm ends at high altitude
        where_end = np.searchsorted(h, self.atm_end)

        if use_telemetry:
            raw_telemetry = pickle.load(open(self._paths['telemetry'], 'rb'))
            telemetry = utils.process_telemetry(raw_telemetry)

            # this function returns dict of telemetry medians within
            # 30 mins of each forecast datapoint and datetimes of these
            tel_m, dates_m = utils.match_telemetry(telemetry, forecast.index)
            self.data_gl = pd.DataFrame(data=tel_m, index=dates_m)
            # only keep forecasts with matching telemetry dates
            forecast = forecast.loc[dates_m]
        else:
            # interpolate forecasts to bit above GL (by ~weather tower height)
            gl = {}
            for k in ['u', 'v', 't', 'speed', 'phi']:
                gl[k] = np.array([utils.interpolate(h, f, 
                                                    self.h0 + .05, ddz=False) 
                                  for f in forecast.loc[k].values])
            self.data_gl = pd.DataFrame(data=gl, index=forecast.index)
        
        # how many datapoints we have now after selections
        self.N = len(forecast)

        # save all forecast data only between ground and upper bound
        self.h = h[where_h0:where_end]
        self.p = p_and_h[src]['p'][::-1][where_h0:where_end]

        for k in ['u', 'v', 't', 'speed', 'phi']:
            forecast[k] = [forecast[k].values[i][where_h0:where_end] 
                           for i in range(self.N)]
        self.data_fa = forecast

        # ground layer ends at ~1km above telescope, save for later 
        self.fa_start = np.searchsorted(self.h, self.h0 + self.gl_height)
        
    def get_measurements(self, pt):
        """Get a matched set of measurements from datapoint with index ``pt``.

        Parameters:
            pt:     Pandas Timestamp of date/time for desired output.

        Returns:
            params: Dictionary of wind and temperature measurements, made of
                    matched ground level and free atmosphere data, at Index 
                    location ``pt``. Keys are [``u``, ``v``, ``speed``, ``phi``,
                    ``t``, ``h``] (see class Attributes for details).
        """
        try:
            gl = self.data_gl.loc[pt]
            fa = self.data_fa.loc[pt]
        except KeyError:
            if type(pt) == str:
                raise TypeError(f'pt type must be pd.Timestamp not str!')
            if type(pt) == pd.Timestamp:
                raise KeyError(f'{pt} not found in data index!')

        direction = np.hstack([gl.at['phi'], fa.at['phi']])

        return {'u': np.hstack([gl.at['u'], fa.at['u']]),
                'v': np.hstack([gl.at['v'], fa.at['v']]),
                'speed': np.hstack([gl.at['speed'], fa.at['speed']]),
                't': np.hstack([gl.at['t'], fa.at['t']]),
                'h': np.hstack([self.h0, self.h]),
                'phi': utils.smooth_dir(direction)}

    def _interpolate(self, p_dict, h_out, s=None):
        """Get interpolations & derivatives of p_dict at new heights h_out."""
        # check h_out values
        if max(h_out) > 100:
            # 100km would be an outrageously high altitude, probably unit error
            raise ValueError('Units of h should be in km, not m.')
        if min(h_out) < min(self.h) or max(h_out) > max(self.h):
            raise ValueError(f"Can't interpolate below {min(self.h)}" + \
                             f" or above {max(self.h)}.")

        # if 'h' in the dictionary, use those heights
        if 'h' in p_dict.keys():
            h_in = p_dict['h']
        # otherwise inputs are raw FA values so use self.h
        else:
            h_in = self.h

        out = {}
        # Note: multipying by 1000 (to m) for derivative units.
        for k in ['u', 'v', 't']:
            out[k], out[f'd{k}dz'] = utils.interpolate(h_in * 1000,
                                                       p_dict[k],
                                                       h_out * 1000,
                                                       s=s)

        # special case (h is always self.h here)
        out['p'], out['dpdz'] = utils.interpolate(self.h * 1000,
                                                  self.p,
                                                  h_out * 1000,
                                                  s=s)
        out['phi'] = utils.smooth_dir(utils.to_direction(out['u'], out['v']))
        out['speed'] = np.hypot(out['u'], out['v'])
        out['h'] = h_out

        return out

    def _draw_j(self, pt=None):
        """Draw values for ground and free atmosphere turbulence."""
        a = 10**(-13)  # because PDFs are in units of 10^-13 m^1/3!
        if self.rho_jv == 0:
            return (self.j_pdf['fa'].rvs(random_state=self._rng) * a,
                    self.j_pdf['gl'].rvs(random_state=self._rng) * a)
        else:
            # assumes pt is a date, pick corresponding GL integral
            return (self.j_pdf['fa'].rvs(random_state=self._rng) * a,
                    self.data_gl.at[pt, 'j_gl'] * a)

    def _get_fa_cn2(self, pt):
        """Get free atmosphere $C_n^2$ and h arrays for datapoint with index pt.

        Empirical model for Cn2 from Osborn et al 2018:
        https://doi.org/10.1093/mnras/sty1898.

        """
        # pick out relevant wind data
        try:
            fa = dict(self.data_fa.loc[pt])
        except KeyError:
            if type(pt) == str:
                raise TypeError(f'pt type must be pd.Timestamp not str!')
            if type(pt) == pd.Timestamp:
                raise KeyError(f'{pt} not found in data index!')
        # only calculate Cn2 with this model starting at FA start
        h_complete = np.linspace(self.h[self.fa_start], max(self.h), 500)
        inputs = self._interpolate(fa, h_complete)
        cn2_complete = utils.osborn(inputs)

        return cn2_complete, h_complete

    def get_turbulence_integral(self, pt, nl, location='mean'):
        """Get turbulence integral at nl layers for given time.

        Parameters:
            pt:         Pandas Timestamp of date/time for desired output.
            nl:         Number of output layers desired.
            location:   Method for setting centroid of layer, must be one of
                        [``mean``, ``com``]. [default: ``mean``]

        Returns:
            j_layers:   Turbulence integrals J, in m^(1/3), for each layer.
            h_layers:   Altitude centroid for each layer, in km.
            h_edges:    Layer boundaries, the integration region corresponding
                        to each ``j``.
        """
        if 'mean' in location:
            h_operation = lambda x, w: np.mean(x)
        elif 'com' in location:
            h_operation = lambda x, w: np.average(x, weights=w)
        else:
            raise ValueError('Layer location not valid!')

        cn2, h_km = self._get_fa_cn2(pt)
        # nl - 1 layers in FA
        b = int(len(h_km) / (nl - 1))
        fa_edges = [h_km[i * b] for i in range(nl-1)] + [h_km[-1]]
        fa_layers = [h_operation(h_km[i*b:(i+1)*b], cn2[i*b:(i+1)*b])
                     for i in range(nl - 1)]
        fa_j_uncal = utils.integrate_in_bins(cn2, h_km, np.array(fa_edges))
        
        # draw Gl and FA total turbulence integral values from PDFs:
        j_fa, j_gl = self._draw_j(pt=pt)

        # total FA value scales the FA weights from integrated Osborn model
        fa_j_cal = [w * j_fa / np.sum(fa_j_uncal) for w in fa_j_uncal]
        
        # return integrated turbulence and layer information
        return (np.array([j_gl] + fa_j_cal),
                np.array([self.h0] + fa_layers),
                np.array([self.h0] + fa_edges))

    def get_parameters(self, pt, nl=8, s=0, location='mean', skycoord=False,
                       alt=None, az=None, lat=30.2446, lon=70.7494):
        """Get parameters from dataset ``pt`` for a set of atmospheric layers.
        
        Returns a dictionary of wind and turbulence parameter dict for ``nl``
        layers, located at altitudes``h`` and with boundaries ``edges``. Each
        atmospheric layer has a turbulence weight ``j`` and wind parameters
        given by east/northward components ``u`` and ``v``, or equivalently by
        ``speed`` and ``phi``, the direction of wind origin measured in degrees
        east of north.

        Parameters:
            pt:         Pandas Timestamp of date/time for desired output.
            nl:         Number of output layers desired. [default: 8]
            s:          Smoothing factor for scipy interpolate. Use 0 for 
                        perfect interpolation, or None for scipy's' best 
                        estimate. [default: 0]
            location:   Method for setting centroid of layer, must be one of
                        (``mean``, ``com``). [default: ``mean``]
            skycoord:   Whether to return parameters in sky coordinates (e.g. if
                        using GalSim, select True). If False, parameters are 
                        returned in local north/east coordinates. If True,
                        values of alt/az must be given. [default: False]
            alt, az:    Altitude and azimuth of telescope pointing. [default:
                        None, None]
            lat, lon:   Latitude and longitude of the observatory, in degrees.
                        [default: Rubin Observatory, at 30.2446, 70.7494]

        """
        try:
            fa = dict(self.data_fa.loc[pt])
        except KeyError:
            if type(pt) == str:
                raise TypeError(f'pt type must be pd.Timestamp not str!')
            if type(pt) == pd.Timestamp:
                raise KeyError(f'{pt} not found in data index!')

        j, h_layers, h_edges = self.get_turbulence_integral(pt=pt, nl=nl,
                                                            location=location)
        fa_params = self._interpolate(fa, h_layers[1:], s=s)
        params = {}

        # stack GL with FA interpolation results for each parameter
        for k in ['u', 'v', 't', 'speed', 'phi']:
            params[k] = np.hstack([self.data_gl.at[pt, k], fa_params[k]])

        params['h'] = h_layers
        params['j'] = j
        params['edges'] = h_edges

        if skycoord == True:
            params = utils.convert_to_galsim(params, alt, az, lat, lon)
        return params

    def draw_datapoint(self):
        """Draw a random datapoint from the dataset and return index."""
        return self._rng.choice(self.data_fa.index)
        
    def draw_parameters(self, nl=8, s=0, location='mean'):
        """Draw a random datapoint from the dataset, and return parameters.

        Output is the same as ``get_parameters``.

        """
        pt = self.draw_datapoint()
        return self.get_parameters(pt=pt, nl=nl, s=s, location=location)

