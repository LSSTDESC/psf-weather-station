"""Class to generate realistic input parameters for atmospheric PSF sims."""

import numpy as np
import pickle
import pandas as pd
import pathlib
from psfws import utils


class ParameterGenerator():
    """Class to generate realistic input parameters for atmospheric PSF sims.

    This class uses two inputs: NOAA Global Forecasting System (GFS) outputs
    and local wind measurements at the observatory. The repo already contains
    these data for Cerro Pachon, and all defaults are set up to match this
    location. Use of the code to generate parameters for Cerro Pachon (and
    Cerro Telolo, nearby) is straightforward, but for use at other
    observatories, users must supply input data: instructions
    for downloading NOAA datasets and formatting telemetry are in the
    repository README.

    Attributes
    ----------
    data_fa : pandas dataframe
        Free atmosphere (>1km above ground) forecasting data, with DateTimes as
        index and columns 'u', 'v', 'speed', 'dir', 't', and 'p'. Each entry is
        a ndarray of values for each altitude, with speed/velocity components
        in m/s, directions in degrees, temperatures in Kelvin, and pressures in
        mbar. The u/v components of velocity correspond to north/south winds,
        respectively, and the wind direction is given as degrees west of north.
    data_gl : pandas dataframe
        Ground layer data, with DateTimes as index and columns 'speed','dir',
        'u', 'v', 't', and optionally 'j_gl' (see rho_jv below). These values
        are temporally matched to data_fa, so have identical indicies.
    h_fa : ndarray
        Altitudes of free atmopshere forecasting data, in km.
    h0 : float
        Altitude of observatory, in km.
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
    get_raw_wind(pt)
        Get a matched set of wind measurements from datapoint with index pt.

    get_wind_interpolation(pt, h_out, kind='gp')
        Get set of wind measurements from datapoint with index pt, interpolated
        to new altitudes h_out. Interpolation type can be specified with the
        'kind' keyword (str, either 'cubic' or 'gp' for Gaussian Process).

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

    def __init__(self, location='cerro-pachon', seed=None,
                 date_range=['2019-05-01', '2019-10-31'],
                 gfs_file='data/gfs_-30.0_289.5_20190501-20191101.pkl',
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

        telemetry_file : str
            Path to file of telemetry data (default is
            'data/tel_dict_CP_20190501-20191101.pkl').

        gfs_file : str
            Path to file of NOAA GFS data (default is
            'data/gfswinds_cp_20190501-20191031.pkl').

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
        # set up the paths to data files, and check they exist.
        psfws_base = pathlib.Path(__file__).parents[0].absolute()

        self._file_paths = \
            {'gfs_data': pathlib.Path.joinpath(psfws_base, gfs_file),
             'telemetry': pathlib.Path.joinpath(psfws_base, telemetry_file)}

        for file_path in self._file_paths.values():
            if not file_path.is_file():
                print(f'code running from: {psfws_base}')
                raise FileNotFoundError(f'file {file_path} not found!')

        # set up random number generator with seed, if given
        self._rng = np.random.default_rng(seed)

        # set ground + telescope height, turbulence pdf (location specific)
        self.h0, self.j_pdf = utils.initialize_location(location)
        # TO DO: put this rho in the location specific utils?
        self.rho_jv = rho_j_wind

        # load and match GFS/telemetry data
        self._load_data(date_range)

        # if using correlation between wind speed and ground turbulence,
        # draw values in advance and perform correlation of marginals
        if self.rho_jv is not None:
            # draw JGL values
            j_gl = self.j_pdf['gl'].rvs(size=self.N, random_state=self._rng)
            # correlate and store modified dataframe
            self.data_gl = utils.correlate_marginals(self.data_gl,
                                                     j_gl,
                                                     self.rho_jv,
                                                     self._rng)

    def _load_data(self, dr=['2019-05-01', '2019-10-31']):
        """Load data from GFS and telemetry pickle files, match, and store."""
        gfs = pickle.load(open(self._file_paths['gfs_data'], 'rb'))
        gfs, h_fa = utils.process_gfs(gfs)

        # set index for lowest GFS data to use according to observatory height:
        # don't use anything lower than 1km above ground
        self._fa_stop = max([10, np.where(h_fa > self.h0 + 1)[0][0]])
        self.h_fa = h_fa[self._fa_stop:]

        # first, find GFS dates within the date range desired
        gfs_dates = gfs[dr[0]:dr[1]].index

        # TO DO: wrap following in if statement if using telemetry

        raw_telemetry = pickle.load(open(self._file_paths['telemetry'], 'rb'))
        telemetry = utils.process_telemetry(raw_telemetry)

        # this function returns dict of telemetry medians within
        # 30 mins of each GFS datapoint and datetimes of these
        tel_m, dates_m = utils.match_telemetry(telemetry, gfs_dates)

        # store results
        # TO DO: or if using gfs, select just the ground layer
        self.data_gl = pd.DataFrame(data=tel_m, index=dates_m)

        gfs = gfs.loc[dates_m]
        self.N = len(gfs)

        # FA data a bit more tricky... just want to keep >1km from ground?
        for k in ['u', 'v', 't', 'p', 'speed', 'dir']:
            gfs[k] = [gfs[k].values[i][self._fa_stop:] for i in range(self.N)]
        self.data_fa = gfs

    def get_raw_measurements(self, pt):
        """Get a matched set of measurements from datapoint with index pt.

        Parameters
        ----------
        pt : int
            date of output datapoint desired

        Returns
        -------
        dict of wind and temperature measurements, made of paired telemetry and
        GFS data for integer index pt.
        Keys are 'u' and 'v' for arrays of velocity components, and 'speed' and
        'direction', 'temp' for temperatures, and 'h' gives array of altitudes
        for all measurements.
        The u/v components of velocity correspond to north/south winds,
        respectively, and the wind direction is given as degrees west of north.

        """
        gl = self.data_gl.loc[pt]
        fa = self.data_fa.loc[pt]

        direction = np.hstack([gl.at['dir'], fa.at['dir']])

        return {'u': np.hstack([gl.at['u'], fa.at['u']]),
                'v': np.hstack([gl.at['v'], fa.at['v']]),
                'speed': np.hstack([gl.at['speed'], fa.at['speed']]),
                't': np.hstack([gl.at['t'], fa.at['t']]),
                'h': np.hstack([self.h0, self.h_fa]),
                'direction': utils.smooth_direction(direction),
                'p': fa.at['p']}

    def _interpolate(self, p_dict, h_out, s=None):
        """Get interpolations & derivatives of params at new heights h_out."""
        # Note: multipying everything by 1000 (to m) for unit consistency.
        out = {}
        for k in ['u', 'v', 't']:
            out[k], out[f'd{k}dz'] = utils.interpolate(p_dict['h'] * 1000,
                                                       p_dict[k],
                                                       h_out * 1000,
                                                       s=s)

        # special case:
        out['p'], out['dpdz'] = utils.interpolate(self.h_fa * 1000,
                                                  p_dict['p'],
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
            # TO DO: add a check for pt being a date
            # as written, assumes pt is a date, pick corresponding GL integral
            return (self.j_pdf['fa'].rvs(random_state=self._rng) * a,
                    self.data_gl.at[pt, 'j_gl'] * a)

    def get_fa_cn2(self, pt):
        """Get free atmosphere Cn2 and h arrays for datapoint with index pt.

        Empirical model for Cn2 from Osborn et al 2018:
        https://doi.org/10.1093/mnras/sty1898.

        """
        # pick out relevant wind data
        raw_p = self.get_raw_measurements(pt)

        # only calculate Cn2 with this model starting 1km above ground
        h_complete = np.linspace(self.h0 + 1, max(self.h_fa), 500)
        inputs = self._interpolate(raw_p, h_complete)
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
        h_interp = np.linspace(self.h0, max(self.h_fa), 500)

        # interpolate the median speeds from GFS to find height of max
        all_speeds = [i for i in self.data_fa['speed'].values]
        h_maxspd = utils.find_max_median(all_speeds, self.h_fa,
                                         h_interp, self.h0)

        # interpolate the median cn2 to find height of max
        # don't change k here because don't care about absolute amplitude
        cn2, h_cn2 = self.get_cn2_all()
        h_maxcn2 = utils.find_max_median(cn2, h_cn2, h_interp, self.h0)

        # sort the heights of max speed and max turbulence
        h3, h4 = np.sort([h_maxspd, h_maxcn2])

        # raise the lowest layer slightly off of the ground
        lowest = self.h0 + 0.250

        h2 = np.mean([lowest, h3])
        h5 = np.mean([h4, 18])

        return [lowest, h2, h3, h4, h5, 18]

    def _integrate_cn2(self, pt, layers='auto'):
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
        maxh = max(self.h_fa)
        cn2, h = self.get_fa_cn2(pt)

        # define bins according to layers argument
        if layers == 'auto':
            bin_centers = self._get_auto_layers(pt)
        else:
            bin_centers = layers
        edges = [self.h0] + [np.mean(bin_centers[i:i+2])
                             for i in range(len(bin_centers)-1)] + [maxh]

        # integrate cn2 in bins defined by these edges
        j = utils.integrate_in_bins(cn2, h, edges)
        # return along with edges and bin centers
        return j, np.array(edges), np.array(bin_centers)

    def get_turbulence_integral(self, pt, layers='auto'):
        """Return integrated Cn2 profile for datapoint with index pt."""
        # draw turbulence integral values from PDFs:
        j_fa, j_gl = self._draw_j(pt=pt)

        fa_ws, _, fa_layers = self._integrate_cn2(pt, layers=layers)
        # total FA value scales the FA weights from integrated Osborn model
        fa_ws = [w * j_fa / np.sum(fa_ws) for w in fa_ws]

        # return integrated turbulence
        return [j_gl] + fa_ws, [self.h0] + fa_layers

    def get_param_interpolation(self, pt, h_out, s=None):
        """Return winds for dataset with index pt interpolated to h_out."""
        p_dict = self.get_raw_measurements(pt)
        return self._interpolate(p_dict, h_out, s)

    def draw_parameters(self, layers='auto', s=None):
        """Draw a random, full set of parameters.

        Parameters: layers sets output altitudes; either array of values, or
        'auto' for them to be automatically calculated based on wind and
        turbulence maximums (default: 'auto').

        Returns: dict of parameters. Keys are 'j' for turbulence integrals,
        'u','v','speed','direction' for wind parameters, and 'h' for altitudes.
        The u/v components of velocity correspond to north/south winds,
        respectively, and the wind direction is given as degrees west of north.
        The turbulence integrals have dimension m^[1/3].
        """
        pt = self._rng.choice(self.data_fa.index)

        j, layers = self.get_turbulence_integral(pt, layers='auto')
        params = self.get_param_interpolation(pt, layers, s=s)

        params['j'] = j

        return params
