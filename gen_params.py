import numpy as np
import pickle
import pandas as pd

class ParameterGenerator():
    '''
    class object to generate realistic atmospheric input parameters required for GalSim simulation.
    uses NOAA GFS predictions matched with telemetry from Cerro Pachon.
    '''
    def __init__(self):

        # False until data have been temporally matched, to keep track of what is being used
        self.matched = False 

    def _load_gfs(self, gfs_path):
        '''
        load pickle file of GFS winds and store in class
        '''

    def _load_telemetry(self, telemetry_path):
        '''
        load pickle file of CP telemetry and store only winds in class
        '''

    def _match_data(self):
        '''
        temporally match the GFS and telemetry data
        '''


        self.matched = True

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



