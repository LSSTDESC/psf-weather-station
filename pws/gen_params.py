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
                 telemetry_file='data/cptelemetry_20190501-20191101.pkl'):
        # define path using pathlib
        self.p = pathlib.Path('.').resolve()
        
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
        self.gfs_stop = 11
        self.cp_ground = 2715 + 50

    def _load_data(self):
        '''
        load pickle file of GFS winds and store in class
        '''
        gfs_winds = pickle.load(open(self.gfs_f, 'rb'))
        gfs_winds = utils.process_gfs(gfs_winds)

        self.gfs_h = np.load(open(self.gfs_h_f, 'rb'))[::-1]

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

        matched_s, matched_d = utils.match_telemetry(speed, directions, gfs_dates)
        
        self.telemetry = {'speed': matched_s,'dir': matched_d}
        self.gfs_winds = gfs_winds.loc[gfs_dates].iloc[ids_to_keep]

    def _construct_wind_set(self):
        '''
        construct a vector of winds vs altitude using GFS till self.gfs_stop and matched telemetry 
        '''



    def _interpolate_wind(self, h):
        '''
        use matched data to interpolate 
        '''

        # utils.interpolate()

    def _calculate_cn2(self):
        '''
        use GFS winds and other models to calculate a Cn2 profile
        '''

    def draw_parameters(self):
        '''
        randomly sample the data
        '''

    # should be a method where you can choose which parameters to draw?
    # should the Cn2 be calculated for all the GFS data or only for when you draw parameters?
    # keyword argument to specify?



