import numpy as np
import pickle
import pandas as pd
import pathlib
from pws import utils

class ParameterGenerator():
    '''
    class object to generate realistic atmospheric input parameters required for GalSim simulation.
    uses NOAA GFS predictions matched with telemetry from Cerro Pachon.
    '''
    def __init__(self, gfs_file='data/gfswinds_cp_20190501-20191031.pkl', gfs_h_file='data/H.npy',
                 telemetry_file='data/cptelemetry_20190501-20191101.pkl', 
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
        self.cp_ground = (2715 + 50) / 1000

        self._match_data()

    def _load_data(self):
        '''
        load pickle file of GFS winds and store in class
        '''
        gfs_winds = pickle.load(open(self.gfs_f, 'rb'))
        gfs_winds = utils.process_gfs(gfs_winds)

        self.gfs_h = np.load(open(self.gfs_h_f, 'rb'))[::-1] / 1000

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

    def _calculate_cn2(self):
        '''
        use GFS winds and other models to calculate a Cn2 profile
        '''

    def get_wind_parameters(self, pt):
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

        height = np.hstack([self.cp_ground, self.gfs_h[self.gfs_stop:]])

        return {'u': u, 'v': v, 'speed': speed, 
                'direction': utils.smooth_direction(direction), 'h': height}

    def draw_wind_parameters(self):
        '''
        randomly sample from the data
        '''
        pt = np.random.choice(range(len(self.gfs_winds)))
 
        return self.get_wind_parameters(pt)

    def do_interpolation(self, p_dict, h_interp, kind='gp'):
        '''
        use matched data to interpolate 
        '''
        new_u = utils.interpolate(p_dict['h'], p_dict['u'], h_interp, kind)
        new_v = utils.interpolate(p_dict['h'], p_dict['v'], h_interp, kind)

        new_direction = utils.to_direction(new_v, new_u)
        return {'u': new_u, 'v': new_v, 'speed': np.hypot(new_u, new_v), 
                'direction': utils.smooth_direction(new_direction), 'h': h_interp}


