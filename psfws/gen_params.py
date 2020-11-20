import numpy as np
import pickle
import pandas as pd
import pathlib
from psfws import utils

class ParameterGenerator():
    '''
    class object to generate realistic atmospheric input parameters required for GalSim simulation.
    uses NOAA GFS predictions matched with telemetry from Cerro Pachon.
    '''
    def __init__(self, seed=None, 
                 date_range=['2019-05-01', '2019-10-31'],
                 gfs_file='data/gfswinds_cp_20190501-20191031.pkl', 
                 gfs_h_file='data/H.npy',
                 telemetry_file='data/tel_dict_CP_20190501-20191101.pkl', 
                 pkg_home='/Users/clairealice/Documents/repos/psf-weather-station'):
        # define path using pathlib -- is there a way to not do this manually? 
        # not sure what I'm looking for exactly 
        self.p = pathlib.Path(pkg_home)
        
        # check that the pointed data files exist:
        self.gfs_f = self.p / pathlib.Path(gfs_file)
        if not self.gfs_f.is_file():
            raise FileNotFoundError(f'file {self.gfs_f} not found!')

        self.gfs_h_f = self.p / pathlib.Path(gfs_h_file)
        if not self.gfs_h_f.is_file():
            raise FileNotFoundError(f'file {self.gfs_h_f} not found!')

        self.tel_f = self.p / pathlib.Path(telemetry_file)
        if not self.tel_f.is_file():
            raise FileNotFoundError(f'file {self.tel_f} not found!')

        # altitude info
        self.gfs_stop = 10
        self.h0 = (2715 + 50) / 1000

        self.rng = np.random.default_rng(seed)

        self._match_data(date_range)

    def _load_data(self):
        '''
        load pickle file of GFS winds and store in class
        '''
        gfs_winds = pickle.load(open(self.gfs_f, 'rb'))
        gfs_winds = utils.process_gfs(gfs_winds)

        self.h_gfs = np.load(open(self.gfs_h_f, 'rb'))[::-1] / 1000

        telemetry = pickle.load(open(self.tel_f, 'rb'))
        telemetry, tel_masks = utils.process_telemetry(telemetry)

        return gfs_winds, telemetry, tel_masks

    def _match_data(self, dr=['2019-05-01', '2019-10-31']):
        '''
        temporally match the GFS and telemetry data
        '''
        gfs_winds, telemetry, tel_masks= self._load_data()

        gfs_dates = gfs_winds[dr[0]:dr[1]].index
        n_gfs = len(gfs_dates)
        
        speed = telemetry['speed'].loc[tel_masks['speed']][dr[0]:dr[1]]
        direction = telemetry['dir'].loc[tel_masks['dir']][dr[0]:dr[1]]

        matched_s, matched_d, updated_gfs_dates = utils.match_telemetry(speed, direction, gfs_dates)
        matched_comps = utils.to_components(matched_s, matched_d)
        
        self.telemetry = {'speed': matched_s, 'dir': matched_d, 
                          'u': matched_comps['u'], 'v': matched_comps['v']}
        self.gfs_winds = gfs_winds.loc[updated_gfs_dates]

        self.N = len(self.gfs_winds)

    def get_raw_wind(self, pt):
        '''
        construct a vector of wind speed+direction vs altitude using GFS,
        till self.gfs_stop and matched telemetry 
        '''
        speed = np.hstack([self.telemetry['speed'][pt],
                           self.gfs_winds.iloc[pt]['speed'][self.gfs_stop:]])

        direction = np.hstack([self.telemetry['dir'][pt],
                               self.gfs_winds.iloc[pt]['dir'][self.gfs_stop:]])

        u = np.hstack([self.telemetry['u'][pt],
                       self.gfs_winds.iloc[pt]['u'][self.gfs_stop:]])

        v = np.hstack([self.telemetry['v'][pt],
                       self.gfs_winds.iloc[pt]['v'][self.gfs_stop:]])

        height = np.hstack([self.h0, self.h_gfs[self.gfs_stop:]])

        return {'u': u, 'v': v, 'speed': speed, 
                'direction': utils.smooth_direction(direction), 'h': height}

    def interpolate_wind(self, p_dict, h_out, kind='gp'):
        '''
        use matched data to interpolate 
        '''
        new_u = utils.interpolate(p_dict['h'], p_dict['u'], h_out, kind)
        new_v = utils.interpolate(p_dict['h'], p_dict['v'], h_out, kind)

        new_direction = utils.to_direction(new_v, new_u)
        return {'u': new_u, 'v': new_v, 'speed': np.hypot(new_u, new_v), 
                'direction': utils.smooth_direction(new_direction), 'h': h_out}

    def get_cn2(self, pt):
        '''
        use GFS winds and other models to calculate a Cn2 profile
        output is stacked Cn2 of hufnagel model and ground layer model
        '''
        huf_stop = 12
        # find the z and wind speed that are valid for the hufnagel model
        h_huf = self.h_gfs[huf_stop:] * 1000
        huf_wind = self.gfs_winds.iloc[pt]['speed'][huf_stop:]
        # calculate hufnagel cn2 for those
        huf = utils.hufnagel(h_huf, huf_wind)
        # get the ground layer cn2
        gl, gl_h = utils.gl_cn2()

        # return stacked hufnagel and ground layer profiles/altitudes
        return np.hstack([gl, huf]), np.hstack([gl_h, h_huf / 1000])

    def get_cn2_all(self):
        '''get array of cn2 values for the whole dataset'''
        cn2_list = []
        for i in range(self.N):
            cn2, h = self.get_cn2(i)
            cn2_list.append(cn2)
        return np.array(cn2_list)

    def _get_auto_layers(self):
        '''calculate placement of layers in the atmosphere according to the ground, 
        max wind speed, max turbulence, etc'''
        maxh = 18.
        temp_h = np.linspace(self.h0, max(self.h_gfs), 1000)

        # interpolate the median speeds from GFS to find height of max
        median_spd = np.median([i for i in self.gfs_winds['speed'].values], axis=0)
        median_spd_smooth = utils.interpolate(self.h_gfs, median_spd, temp_h, kind='cubic')
        h_max_spd = temp_h[np.argmax(median_spd_smooth)]

        # interpolate the median cn2 to find height of max
        median_cn2 = np.median(self.get_cn2_all(), axis=0)
        _, cn2_h = self.get_cn2(1)
        median_cn2_smooth = utils.interpolate(cn2_h, median_cn2, temp_h, kind='cubic')
        h_max_cn2 = temp_h[100:][np.argmax(median_cn2_smooth[100:])] # find max in mid-atm, not ground

        # sort the heights of max speed and max turbulence
        h3, h4 = np.sort([h_max_spd, h_max_cn2])

        # raise the lowest layer slightly off of the ground
        lowest = self.h0 + 0.250

        return [lowest, np.mean([lowest,h3]), h3, h4, np.mean([h4,maxh]), maxh]

    def integrate_cn2(self, cn2, h, layers='auto'):
        '''
        get an integrated Cn2 profile 
        :layers: how to define the bins for the integration. 'er' to follow the bin centers from ER 2000,
        '''
        maxh = max(self.h_gfs)
        
        # define bins according to layers argument   
        if layers == 'auto':
            bin_centers = self._get_auto_layers()
        elif layers == 'er':
            bin_centers = [i+self.h0 for i in [0,1.8,3.3, 5.8,7.4,13.1,15.8]]
        elif type(layers) == list:
            bin_centers = layers
        edges = [self.h0]+[np.mean(bin_centers[i:i+2]) for i in range(len(bin_centers)-1)]+[maxh]

        # integrate cn2 in bins defined by these edges
        j = utils.integrate_in_bins(cn2, h, edges)
        # return along with edges and bin centers
        return j, np.array(edges), np.array(bin_centers)

    def get_turbulence_integral(self, pt, layers='auto'):
        '''
        get an integrated Cn2 profile for dataset pt
        '''
        cn2, h = self.get_cn2(pt)

        return self.integrate_cn2(cn2, h, layers=layers)

    def get_wind_interpolation(self, pt, h_out, kind='gp'):
        '''return interpolated winds for a dataset'''
        wind_dict = self.get_raw_wind(pt)
        return self.interpolate_wind(wind_dict, h_out, kind=kind)

    def draw_parameters(self, layers='auto'):
        '''draw a random, full set of parameters. 
        returns a dict of layers, wind params, and turbulence integrals '''
        pt = self.rng.integers(low=0,high=len(self.gfs_winds))

        j, _, layers = self.get_turbulence_integral(pt, layers='auto')
        params = self.get_wind_interpolation(pt, layers)

        params['j'] = j

        return params




