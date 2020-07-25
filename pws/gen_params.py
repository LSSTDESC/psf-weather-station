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
    def __init__(self):

        # False until data have been temporally matched, to keep track of what is being used
        self.matched = False 
        # define path using pathlib
        self.p = pathlib.Path('.').resolve()
        self.gfs_loaded = False
        self.cp_loaded = False

        self.gfs_ground = 6

    def _load_gfs(self, gfs_path='data/gfswinds_cp_20190501-20191031.pkl'):
        '''
        load pickle file of GFS winds and store in class
        '''
        gfs_path = self.p / pathlib.Path(gfs_path)

        if gfs_path.is_file():
            gfs_winds = pickle.load(open(gfs_path, 'rb'))
            self.gfs_winds = utils.process_gfs(gfs_winds)
        else:
            raise FileNotFoundError(f'file {gfs_path} not found!')

        h_path = gfs_path.with_name('H.npy')
        if h_path.is_file():
            self.gfs_h = np.load(open(h_path, 'rb'))[::-1]
        else:
            raise FileNotFoundError(f'file {h_path} not found!')

        return True

    def _load_telemetry(self, telemetry_path='data/cptelemetry_20190501-20191101.pkl'):
        '''
        load pickle file of CP telemetry and store only winds in class
        '''
        telemetry_path = self.p / pathlib.Path(telemetry_path)

        if telemetry_path.is_file():
            telemetry = pickle.load(open(telemetry_path, 'rb'))

            self.cp_telemetry, self.cp_masks = utils.process_telemetry(telemetry)
            self.cp_ground = 2715 + 50
        else:
            raise FileNotFoundError(f'file {telemetry_path} not found!')

        return True

    def _match_data(self, dr=['2019-05-01', '2019-10-31']):
        '''
        temporally match the GFS and telemetry data
        '''
        if self.gfs_loaded and self.cp_loaded:
            print
            gfs_dates = self.gfs_winds['speed'][dr[0]:dr[1]].index
            n_gfs = len(gfs_dates)
            cp_in_range = self.cp_telemetry['speed'][self.cp_masks['speed']][dr[0]:dr[1]]
            
            cp_matched = np.zeros(n_gfs)
            for i in range(n_gfs):
                cp_ids_close = abs(gfs_dates[i] - cp_in_range.index) < pd.Timedelta('30min')
                cp_close = cp_in_range[cp_ids_close]['vals']
                cp_matched[i] = cp_close[cp_close<40].median()

            self.cp_telemetry['matched'] = cp_matched
            return True
        else:
            return False

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



