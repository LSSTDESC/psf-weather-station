import numpy as np
import pickle
import pandas as pd
import pathlib

class ParameterGenerator():
    '''
    class object to generate realistic atmospheric input parameters required for GalSim simulation.
    uses NOAA GFS predictions matched with telemetry from Cerro Pachon.
    '''
    def __init__(self):

        # False until data have been temporally matched, to keep track of what is being used
        self.matched = False 
        # define path using pathlib
        self.p = pathlib.Path('.')
        self.gfs_loaded = False
        self.cp_loaded = False

        self.gfs_ground = 6


    def _load_gfs(self, gfs_path='/data/gfswinds_20190501-20191031.pkl'):
        '''
        load pickle file of GFS winds and store in class
        '''
        gfs_path = self.p/pathlib.Path(gfs_path)
        if gfs_path.is_file():
            self.gfs_winds = pickle.load(open(gfs_path, 'rb'))
        else:
            raise FileNotFoundError(f'file {gfs_path} not found!')

        self.gfs_winds['speed'] = [np.hypot(gfs_winds['u'].values[i][::-1], 
                                            gfs_winds['v'].values[i][::-1])[self.gfs_ground:] 
                                   for i in range(len(gfs_winds))]
        
        h_path = gfs_path.with_name('H.npy')
        if h_path.is_file():
            self.gfs_h = np.load(open(h_path, 'rb'))
        else:
            raise FileNotFoundError(f'file {h_path} not found!')

        return True

    def _process_gfs(self):
        '''
        Process GFS data: calculate speed and directions
        '''
        if self.gfs_loaded:
            self.gfs_winds['speed'] = [np.hypot(self.gfs_winds['u'].values[i][::-1], 
                                                self.gfs_winds['v'].values[i][::-1])[self.gfs_ground:] 
                                       for i in range(len(self.gfs_winds))]

            self.gfs_winds['dir'] = [utils.smooth_direction(self.gfs_winds['v'].values[i][::-1],
                                                            self.gfs_winds['u'].values[i][::-1])[self.gfs_ground:] 
                                       for i in range(len(self.gfs_winds))]


    def _load_telemetry(self, telemetry_path='/data/cptelemetry_20190501-20191101.pkl'):
        '''
        load pickle file of CP telemetry and store only winds in class
        '''
        telemetry_path = self.p/pathlib.Path(telemetry_path)
        if telemetry_path.is_file():
            telemetry = pickle.load(open(telemetry_path, 'rb'))
        else:
            raise FileNotFoundError(f'file {telemetry_path} not found!')

        self.cp_wind_dir = pd.DataFrame(telemetry['WindDir_twr']) 
        self.cp_wind_speed = pd.DataFrame(telemetry['WindSpd_twr'])

        for dframe in [self.cp_wind_dir, self.cp_wind_speed]:
            dframe.index = pd.to_datetime(dframe['dts'], utc=True)

        self.cp_masks = {'speed': self.cp_wind_speed != 0,
                         'dir': self.cp_wind_dir <= 360}

        return True

    def _match_data(self, dr=['2019-05-01', '2019-10-31']):
        '''
        temporally match the GFS and telemetry data
        '''
        if self.gfs_loaded and self.cp_loaded:
            gfs_in_range = self.gfs_winds[dr[0], dr[1]]
            cp_in_range = self.cp_wind_speed[dr[0], dr[1]][self.cp_masks['speed']]
            
            cp_matched = np.zeros(len(gfs_in_range))
            for i in range(len(gfs_in_range)):
                cp_ids_close = abs(gfs_in_range.index[i] - cp_in_range.index) < pd.Timedelta('30min')
                cp_close = cp_in_range[cp_ids_close]
                cp_matched[i] = cp_close[cp_close<40].median()

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



