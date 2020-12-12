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
    gfs_winds : pandas dataframe
        GFS wind values (matched w telemetry), with DateTime as index and
        columns 'u', 'v', 'speed', 'dir'. Each entry in the DataFrame is a
        ndarray of values for each altitude, with speed/velocity components
        in m/s and directions in degrees. The u/v components of velocity
        correspond to north/south winds, respectively, and the wind direction
        is given as degrees west of north.
    telemetry : dict of ndarrays
        Keys: 'speed','dir', 'u', 'v'. Telemetry data, temporally matched to
        GFS wind data: values in gfs_winds['u'].iloc[i] are in the same time
        bin as telemetry['u'][i].
    h_gfs : ndarray
        Altitudes of GFS datapoints, in km.
    h0 : float
        Altitude of observatory, in km.
    h_dome : float
        Height of telescope dome, in meters.
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

    get_cn2(pt)
        Get a set of Cn2 values associated with datapoint with index pt.

    get_turbulence_integral(pt, layers='auto')
        Get set of turbulence integrals associated with datapoint with index pt
        Centers of integration regions set by layers keyword; either array of
        values, or 'auto' for them to be automatically calculated based on wind
        and turbulence maximums.

    get_cn2_all()
        Get Cn2 values for entire dataset, returned as array.

    draw_parameters(layers='auto')
        Randomly draw a set of parameters: wind speed, wind direction,
        turbulence integral. These are returned in a dict along with layer
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
                 gfs_file='data/gfs_-30.0_289.5_20190501-20191031.pkl',
                 telemetry_file='data/tel_dict_CP_20190501-20191101.pkl'):
        """Initialize generator and process input data.

        Parameters
        ----------
        location : str
            The name of desired mountaintop (default is 'cerro-pachon').
            Valid options: 'cerro-paranal', 'cerro-pachon', 'cerro-telolo',
            'mauna-kea', and 'la-palma'.
            To customize to another observatory, input instead a dict with keys
            'altitude' (value in km) and 'height' (optional: dome height in m.
            If not given, will default to 10m)

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

        """
        # set up the paths to data files, and check they exist.
        psfws_base = pathlib.Path(__file__).parents[1].absolute()

        self._file_paths = \
            {'gfs_data': pathlib.Path.joinpath(psfws_base, gfs_file),
             'gfs_alts': pathlib.Path.joinpath(psfws_base, 'data/H.npy'),
             'telemetry': pathlib.Path.joinpath(psfws_base, telemetry_file)}

        for file_path in self._file_paths.values():
            if not file_path.is_file():
                raise FileNotFoundError(f'file {file_path} not found!')

        # set up random number generator with seed, if given
        self._rng = np.random.default_rng(seed)

        # set ground height and telescope height (location specific)
        self.h0, self.h_dome = utils.initialize_location(location)

        # load and match GFS/telemetry data
        self._match_data(date_range)

        # set index for lowest GFS data to use according to observatory height:
        # don't use anything lower than 1km above ground
        self._gfs_stop = max([10, np.where(self.h_gfs > self.h0 + 1)[0][0]])

    def _load_data(self):
        """Load and return data from GFS and telemetry pickle files."""
        gfs_winds = pickle.load(open(self._file_paths['gfs_data'], 'rb'))
        gfs_winds = utils.process_gfs(gfs_winds)

        # order heights from small to large, and convert to km
        self.h_gfs = np.sort(np.load(open(self._file_paths['gfs_alts'],
                                          'rb'))) / 1000

        raw_telemetry = pickle.load(open(self._file_paths['telemetry'], 'rb'))
        telemetry = utils.process_telemetry(raw_telemetry)

        return gfs_winds, telemetry

    def _match_data(self, dr=['2019-05-01', '2019-10-31']):
        """Load data, temporally match telemetry to GFS, add as attribute."""
        # load in data
        gfs_winds, telemetry = self._load_data()

        # first, find GFS dates within the date range desired
        gfs_dates = gfs_winds[dr[0]:dr[1]].index

        # this function returns median of measured speed/dir telemetry within
        # 30 mins of each GFS datapoint, when possible, and datetimes of these
        spd_m, dir_m, dates_m = utils.match_telemetry(telemetry, gfs_dates)

        # calculate velocity componenents from the matched speed/directions
        uv_m = utils.to_components(spd_m, dir_m)

        # store results
        self.telemetry = {'speed': spd_m, 'dir': dir_m,
                          'u': uv_m['u'], 'v': uv_m['v']}
        self.gfs_winds = gfs_winds.loc[dates_m]

        self.N = len(self.gfs_winds)

    def get_raw_wind(self, pt):
        """Get a matched set of wind measurements from datapoint with index pt.

        Parameters
        ----------
        pt : int
            index of output datapoint desired

        Returns
        -------
        dict of wind measurements, made of paired telemetry and GFS data for
        integer index pt.
        Keys are 'u' and 'v' for arrays of velocity components, 'speed' and
        'direction', and 'h' gives array of altitudes for all measurements.
        The u/v components of velocity correspond to north/south winds,
        respectively, and the wind direction is given as degrees west of north.

        """
        speed = np.hstack([self.telemetry['speed'][pt],
                           self.gfs_winds.iloc[pt]['speed'][self._gfs_stop:]])

        direction = np.hstack([self.telemetry['dir'][pt],
                               self.gfs_winds.iloc[pt]['dir'][self._gfs_stop:]]
                              )

        u = np.hstack([self.telemetry['u'][pt],
                       self.gfs_winds.iloc[pt]['u'][self._gfs_stop:]])

        v = np.hstack([self.telemetry['v'][pt],
                       self.gfs_winds.iloc[pt]['v'][self._gfs_stop:]])

        height = np.hstack([self.h0, self.h_gfs[self._gfs_stop:]])

        return {'u': u, 'v': v, 'speed': speed,
                'direction': utils.smooth_direction(direction), 'h': height}

    def _interpolate_wind(self, p_dict, h_out, kind='gp'):
        """Return new wind dict, vals interpolated to new heights h_out."""
        new_u = utils.interpolate(p_dict['h'], p_dict['u'], h_out, kind)
        new_v = utils.interpolate(p_dict['h'], p_dict['v'], h_out, kind)

        new_direction = utils.to_direction(new_v, new_u)

        return {'u': new_u, 'v': new_v, 'speed': np.hypot(new_u, new_v),
                'direction': utils.smooth_direction(new_direction), 'h': h_out}

    def _draw_ground_cn2(self):
        """Return ground Cn2 model, with parameters randomly drawn.

        Model from "where is surface layer turbulence" (Tokovinin 2010), params
        drawn from normal distribution around values from paper.
        """
        # model valid till 300m
        h_gl = np.linspace(self.h_dome, 300, 100)

        tokovinin_a = [-13.38, -13.15, -13.51]
        tokovinin_p = [-1.16, -1.17, -0.97]

        # adjust by sqrt(N-1) for sample variance
        a = self._rng.normal(loc=np.mean(tokovinin_a),
                             scale=np.std(tokovinin_a)/np.sqrt(2))
        p = self._rng.normal(loc=np.mean(tokovinin_p),
                             scale=np.std(tokovinin_p)/np.sqrt(2))

        # also return h of GL model in km, adjusted to observatory height
        return utils.ground_cn2_model({'A': a, 'p': p},
                                      h_gl), self.h0 + h_gl/1000

    def get_cn2(self, pt):
        """Get Cn2 and h arrays for datapoint with index pt.

        Use Hufnagel model and GFS winds to calculate a Cn2 profile above 3km,
        stacked with a ground layer model draw.
        """
        # pick out relevant wind data
        raw_winds = self.get_raw_wind(pt)

        # make a vector of heights where hufnagel will be valid
        h_huf = np.linspace(self.h0 + 3, max(self.h_gfs), 100)
        # interpolate wind data to those heights
        speed_huf = self._interpolate_wind(raw_winds, h_huf)['speed'].flatten()
        # calculate hufnagel cn2 for those
        cn2_huf = utils.hufnagel(h_huf, speed_huf)

        # draw model of ground turbulence
        cn2_gl, h_gl = self._draw_ground_cn2()

        cn2_complete = np.hstack([cn2_gl, cn2_huf])
        h_complete = np.hstack([h_gl, h_huf])

        # return stacked hufnagel and ground layer profiles/altitudes
        return cn2_complete, h_complete

    def get_cn2_all(self):
        """Get array of Cn2 values for all data available."""
        cn2_list = []
        for i in range(self.N):
            cn2, h = self.get_cn2(i)
            cn2_list.append(cn2)
        return np.array(cn2_list), h

    def _get_auto_layers(self):
        """Return layer altitudes according to max wind speed & turbulence."""
        # make an array of heights for interpolation
        h_interp = np.linspace(self.h0 + self.h_dome, max(self.h_gfs), 500)

        # interpolate the median speeds from GFS to find height of max
        all_speeds = [i for i in self.gfs_winds['speed'].values]
        h_maxspd = utils.find_max_median(all_speeds, self.h_gfs,
                                         h_interp, self.h0)

        # interpolate the median cn2 to find height of max
        all_cn2, h_cn2 = self.get_cn2_all()
        h_maxcn2 = utils.find_max_median(all_cn2, h_cn2, h_interp, self.h0)

        # sort the heights of max speed and max turbulence
        h3, h4 = np.sort([h_maxspd, h_maxcn2])

        # raise the lowest layer slightly off of the ground
        lowest = self.h0 + 0.250

        h2 = np.mean([lowest, h3])
        h5 = np.mean([h4, 18])

        return [lowest, h2, h3, h4, h5, 18]

    def _integrate_cn2(self, cn2, h, layers='auto'):
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
        maxh = max(self.h_gfs)

        # define bins according to layers argument
        if layers == 'auto':
            bin_centers = self._get_auto_layers()
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
        # get cn2 of pt
        cn2, h = self.get_cn2(pt)

        # return integrated cn2
        return self._integrate_cn2(cn2, h, layers=layers)

    def get_wind_interpolation(self, pt, h_out, kind='gp'):
        """Return winds for dataset with index pt interpolated to h_out."""
        wind_dict = self.get_raw_wind(pt)
        return self._interpolate_wind(wind_dict, h_out, kind=kind)

    def draw_parameters(self, layers='auto'):
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
        pt = self._rng.integers(low=0, high=len(self.gfs_winds))

        j, _, layers = self.get_turbulence_integral(pt, layers='auto')
        params = self.get_wind_interpolation(pt, layers)

        params['j'] = j

        return params
