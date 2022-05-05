"""Class to generate realistic input parameters for atmospheric PSF sims."""

import numpy as np
import pickle
import pandas as pd
import pathlib
from psfws import utils


class ParameterGenerator():
    """Class to generate realistic input parameters for atmospheric PSF sims.

    This class uses as main input global circulation model weather forecasting
    outputs, from either the NOAA Global Forecasting System (GFS) analysis or
    the European Center for Midrange Weather Forecasting (ECMWF) reanalysis 
    dataset ERA5.
    Optionally, local wind measurements from the site of interest may be used
    to improve the accuracy of the outputs. The package contains these data
    for Cerro Pachon, and all defaults are set up to match this location.
    Use of the code to generate parameters for Cerro Pachon (and nearby Cerro 
    Telolo, nearby) is straightforward, but for use at other observatories,
    users must supply input data: instructions for downloading and formatting 
    forecasting data/telemetry are in the README.

    Attributes
    ----------
    data_fa : pandas dataframe
        Above ground forecasting data, with DateTimes as index and columns 'u',
        'v', 'speed', 'dir', 't', and 'p'. Each entry is a ndarray of values
        for each altitude, with speed/velocity components in m/s, directions in
        degrees, temperatures in Kelvin, and pressures in mbar. The u/v
        components of velocity correspond to north/south winds, respectively,
        and the wind direction is given as degrees west of north.
        To select data in the free atmosphere use the gl_end parameter, for
        example: data_fa.at[pt,'v'][fa_start:]
    data_gl : pandas dataframe
        Ground layer data, with DateTimes as index and columns 'speed','dir',
        'u', 'v', 't', and optionally 'j_gl' (see rho_jv below). These values
        are temporally matched to data_fa, so have identical indicies.
        The data are either telemetry, if a data file was given, or forecast
        data interpolated to ground altitude.
    h : ndarray
        Altitudes of free atmopshere forecasting data, in km.
    h0 : float
        Altitude of observatory, in km.
    fa_start : int
        Index of h corresponding to the start of the free atmosphere ~1km above 
        ground, to use when selecting for free atmosphere parameters.
    j_pdf : dict
        Dictionary containing parameters for lognormal PDFs of turbulence
        integrals for both the ground layer and the free atmosphere values.
        Keys are 'gl' and 'fa' respecitvely.
    rho_jv : float or None (default)
        Correlation coefficient between the ground layer wind speed and ground
        turbulence integral. If None, no correlation is included.
    N : int
        Number of matched datasets.


    Methods
    -------
    get_raw_measurements(pt)
        Get a matched set of measurements from datapoint with index pt.

    get_param_interpolation(pt, h_out, s=None)
        Get set of parameters from datapoint with index pt, interpolated
        to new altitudes h_out. Smoothness of cubic interpolation can be
        specified with keyword s: None for scipy optimized value, 0 for no 
        smoothing.

    get_fa_cn2(pt)
        Get free atmosphere Cn2 profile for requested datapoint.

    get_turbulence_integral(pt, layers='auto')
        Get set of turbulence integrals associated for requested datapoint pt.
        Centers of integration regions set by layers keyword; either array of
        values, or 'auto' for them to be automatically calculated based on wind
        and turbulence maximums.

    get_cn2_all()
        Get free atmosphere Cn2 profiles for entire dataset, returned as array.

    draw_parameters(layers='auto')
        Randomly draw a set of parameters: wind speed, wind direction,
        turbulence integrals. These are returned in a dict along with layer
        heights.
        Output altitudes are set by layers keyword; either array of values, or
        'auto' for them to be automatically calculated based on wind and
        turbulence maximums.

    Notes
    -----
    Code is written to output quantities formatted to match desired inputs for
    GalSim atompsheric PSF simulations.

    """

    def __init__(self, seed=None, h_tel=2.715, rho_jv=None,
                 turbulence={'gl':{'s':0.62,'scale':2.34},'fa':{'s':0.84,'scale':1.51}},
                 forecast_file='data/ecmwf_-30.25_-70.75_20190501_20191031.p',
                 telemetry_file='data/tel_dict_CP_20190501-20191101.pkl',
                 rho_j_wind=None):
        """Initialize generator and process input data.

        Parameters
        ----------
        location : str
            The name of desired mountaintop (default is 'cerro-pachon').
            Valid options: 'cerro-paranal', 'cerro-pachon', 'cerro-telolo',
            'mauna-kea', and 'la-palma'.
            To customize to another observatory, input instead a dict with keys
            'altitude' (value in km) and 'turbulence_params', itself a nested
            dict of lognormal PDF parameters 's' (sigma) and 'scale' (exp(mu))
            for ground layer and free atmosphere, e.g. {'gl':{'s':, 'scale':}}.

        seed : int
            Seed to initialize random number generator (default is None)

        telemetry_file : str or None
            Path to file of telemetry data (default is
            'data/tel_dict_CP_20190501-20191101.pkl'). If None, forecast data
            will be used for ground layer information.

        forecast_file : str
            Path to file of weather forecast data (default is
            'data/gfs_-30.0_289.5_20190501-20191101.pkl').

        date_range : list
            List of two strings representing dates, e.g. '2019-05-01'.
            Data date range to use. Allows user to select subset of telemetry
            (default: ['2019-05-01', '2019-10-31'])

        rho_j_wind : float (default is None)
            Desired correlation coefficient between ground wind speed and
            turbulence integral. If None, no correlation is included. If a
            float value is specified, the joint PDF of wind values and ground
            turbulence is generated and the turbulence values are stored in
            data_gl as the 'j_gl' column.

        """
                 telemetry_file='data/tel_dict_CP_20190501-20191101.pkl'):
        # set up the paths to data files, and check they exist.
        psfws_base = pathlib.Path(__file__).parents[0].absolute()

        self._paths = \
            {'forecast_data': pathlib.Path.joinpath(psfws_base, forecast_file),
             'p_and_h': pathlib.Path.joinpath(psfws_base, 'data/p_and_h.p')}

        if telemetry_file is not None:
            self._paths['telemetry'] = pathlib.Path.joinpath(psfws_base, 
                                                             telemetry_file)
            use_telemetry = True
        else:
            use_telemetry = False
             
        for file_path in self._paths.values():
            if not file_path.is_file():
                print(f'code running from: {psfws_base}')
                raise FileNotFoundError(f'file {file_path} not found!')

        # set ground altitude, turbulence pdf, and j/wind correlation
        self.h0 = h_tel 
        self.j_pdf = {k: utils.lognorm(v['s'], v['scale']) for k, v in turbulence.items()}
        self.rho_jv = rho_jv

        self._rng = np.random.default_rng(seed)

        # load and match forecast/telemetry data
        self._load_data(use_telemetry)

        # if using correlation between wind speed and ground turbulence,
        # draw values in advance and perform correlation of marginals
        if self.rho_jv is not 0:
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

        try: # first, find forecast dates within the date range desired
            forecast_dates = forecast.index
        except KeyError:
            print("Requested dates are not within range of available data!")

        # load heights and pressures
        p_and_h = pickle.load(open(self._paths['p_and_h'], 'rb'))
        src = 'ecmwf' if len(forecast['u'].iat[0]) > 50 else 'noaa'
        # reverse to match forecast data order, convert to m
        h = np.array([h/1000 for h in p_and_h[src]['h'][::-1]])

        # find lower gl cutoff:
        where_h0 = np.where(h > self.h0)[0][0]
        # free atm ends at high altitude
        where_end = np.where(h > self.h0 + 23)[0][0]

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
            for k in ['u', 'v', 't', 'speed', 'dir']:
                gl[k] = np.array([utils.interpolate(h, f, 
                                                    self.h0 + .05, ddz=False) 
                                  for f in forecast.loc[k].values])
            self.data_gl = pd.DataFrame(data=gl, index=forecast.index)
        
        # how many datapoints we have now after selections
        self.N = len(forecast)

        # save all forecast data only between ground and upper bound
        self.h = h[where_h0:where_end]
        self.p = p_and_h[src]['p'][::-1][where_h0:where_end]

        for k in ['u', 'v', 't', 'speed', 'dir']:
            forecast[k] = [forecast[k].values[i][where_h0:where_end] 
                           for i in range(self.N)]
        self.data_fa = forecast

        # ground layer ends at ~1km above telescope, save for later 
        self.fa_start = np.where(self.h > self.h0 + .8)[0][0]   
        
    def get_measurements(self, pt):
        """Get a matched set of measurements from datapoint with index pt.

        Parameters
        ----------
        pt : int
            pandas Timestamp of date/time of output datapoint desired

        Returns
        -------
        dict of wind and temperature measurements, made of paired telemetry and
        forecast data for integer index pt.
        Keys are 'u' and 'v' for arrays of velocity components, and 'speed' and
        'direction', 'temp' for temperatures, and 'h' gives array of altitudes
        for all measurements.
        The u/v components of velocity correspond to north/south winds,
        respectively, and the wind direction is given as degrees west of north.

        """
        try:
            gl = self.data_gl.loc[pt]
            fa = self.data_fa.loc[pt]
        except KeyError:
            if type(pt) == str:
                raise TypeError(f'pt type must be pd.Timestamp not str!')
            if type(pt) == pd.Timestamp:
                raise KeyError(f'{pt} not found in data index!')

        direction = np.hstack([gl.at['dir'], fa.at['dir']])

        return {'u': np.hstack([gl.at['u'], fa.at['u']]),
                'v': np.hstack([gl.at['v'], fa.at['v']]),
                'speed': np.hstack([gl.at['speed'], fa.at['speed']]),
                't': np.hstack([gl.at['t'], fa.at['t']]),
                'h': np.hstack([self.h0, self.h]),
                'direction': utils.smooth_direction(direction)}

    def _interpolate(self, p_dict, h_out, s=None):
        """Get interpolations & derivatives of params at new heights h_out."""
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
        out['direction'] = utils.smooth_direction(utils.to_direction(out['v'],
                                                                     out['u']))
        out['speed'] = np.hypot(out['u'], out['v'])
        out['h'] = h_out

        return out

    def _draw_j(self, pt=None):
        """Draw values for ground and free atmosphere turbulence."""
        a = 10**(-13)  # because PDFs are in units of 10^-13 m^1/3!
        if self.rho_jv is None:
            return (self.j_pdf['fa'].rvs(random_state=self._rng) * a,
                    self.j_pdf['gl'].rvs(random_state=self._rng) * a)
        else:
            # assumes pt is a date, pick corresponding GL integral
            return (self.j_pdf['fa'].rvs(random_state=self._rng) * a,
                    self.data_gl.at[pt, 'j_gl'] * a)

    def _get_fa_cn2(self, pt):
        """Get free atmosphere Cn2 and h arrays for datapoint with index pt.

        Empirical model for Cn2 from Osborn et al 2018:
        https://doi.org/10.1093/mnras/sty1898.

        """
        # pick out relevant wind data
        fa = dict(self.data_fa.loc[pt])

        # only calculate Cn2 with this model starting at FA start
        h_complete = np.linspace(self.h[self.fa_start], max(self.h), 500)
        inputs = self._interpolate(fa, h_complete)
        cn2_complete = utils.osborn(inputs)

        return cn2_complete, h_complete

    def get_cn2_all(self):
        """Get array of free atmosphere Cn2 values for all data available."""
        cn2_list = []
        for pt in self.data_fa.index:
            cn2, h = self.get_fa_cn2(pt)
            cn2_list.append(cn2)
        return np.array(cn2_list), h

    def _get_auto_layers(self, pt):
        """Return layer altitudes according to max wind speed & turbulence."""
        # make an array of heights for interpolation
        h_interp = np.linspace(self.h0, max(self.h), 500)

        # interpolate the median speeds from forecast to find height of max
        all_speeds = [i for i in self.data_fa['speed'].values]
        h_maxspd = utils.find_max_median(all_speeds, self.h, h_interp, self.h0)

        # interpolate the median cn2 to find height of max
        # don't change k here because don't care about absolute amplitude
        cn2, h_cn2 = self.get_cn2_all()
        h_maxcn2 = utils.find_max_median(cn2, h_cn2, h_interp, self.h0)

        # sort the heights of max speed and max turbulence
        h3, h4 = np.sort([h_maxspd, h_maxcn2])

        # lowest boundary is start of free atmosphere
        lowest = self.h[self.fa_start]

        h2 = np.mean([lowest, h3])
        h5 = np.mean([h4, 18])

        return [h2, h3, h4, h5, 18]

    def _integrate_cn2(self, pt):
        """Calculate turbulence integral of given Cn2 at given layers.

        Parameters
        ----------
        cn2 : array
            cn2 values
        h : array
            heights of cn2 values
        layers : str or list or array
            Centers of integration regions set by layers keyword; either
            list/array of values, or 'auto' for them to be automatically
            calculated based on wind and turbulence maximums (default: 'auto')

        Returns
        -------
        turbulence integral values J, in m^(1/3)
        integration bin edges, in km
        integration bin centers (aka layers if not 'auto'), in km

        """
        maxh = max(self.h)
        cn2, h = self.get_fa_cn2(pt)

        # define bins, lower boundary is start of FA
        bin_centers = self._get_auto_layers(pt)
        edges = [self.h[self.fa_start]] + [np.mean(bin_centers[i:i+2])
                                           for i in range(len(bin_centers)-1)]\
                                           + [maxh]

        # integrate cn2 in bins defined by these edges
        j = utils.integrate_in_bins(cn2, h, edges)
        # return along with edges and bin centers
        return j, edges, bin_centers

    def get_turbulence_integral(self, pt):
        """Return integrated Cn2 profile for datapoint with index pt."""
        # draw turbulence integral values from PDFs:
        j_fa, j_gl = self._draw_j(pt=pt)

        fa_ws, _, fa_layers = self._integrate_cn2(pt)
        # total FA value scales the FA weights from integrated Osborn model
        fa_ws = [w * j_fa / np.sum(fa_ws) for w in fa_ws]
        
        # return integrated turbulence
        return np.array([j_gl] + fa_ws), np.array([self.h0] + fa_layers)

    def get_parameters(self, pt, s=None):
        """Get parameters for dataset pt at automatically generated altitudes.

        Parameters
        ==========
        pt : pd.Timestamp
            timestamp corresponding to dataset of interest. Must be within the
            index of self.data_* 

        Returns
        =======
        parameters : dict
            Keys are 'j' for turbulence integrals, 'u', 'v', 'speed',
            'direction' for wind parameters, and 'h' for altitudes.
            Turbulence integrals have dimension m^[1/3], u/v components of 
            velocity correspond to north/east winds, respectively, and the 
            wind direction is given as degrees west of north.

        """
        try:
            fa = dict(self.data_fa.loc[pt])
        except KeyError:
            if type(pt) == str:
                raise TypeError(f'pt type must be pd.Timestamp not str!')
            if type(pt) == pd.Timestamp:
                raise KeyError(f'{pt} not found in data index!')

        j, h_layers = self.get_turbulence_integral(pt)
        fa_params = self._interpolate(fa, h_layers[1:], s)
        params = {}

        # stack GL with FA interpolation results for each parameter
        for kgl, kfa in zip(['u', 'v', 't', 'speed', 'dir'],
                            ['u', 'v', 't', 'speed', 'direction']):
            params[kfa] = np.hstack([self.data_gl.at[pt, kgl], fa_params[kfa]])

        params['h'] = h_layers
        params['j'] = j
        return params

    def draw_parameters(self, layers='auto', s=None):
        """Draw a random datapoint from the dataset and return parameters."""
        pt = self._rng.choice(self.data_fa.index)
        return self.get_parameters(pt, s)

