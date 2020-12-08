import numpy as np
import pickle
import pandas as pd
import pathlib
from psfws import utils

class ParameterGenerator():
    '''
    class object to generate realistic atmospheric input parameters required for GalSim simulation.
    uses NOAA GFS predictions matched with telemetry from Cerro Pachon.
    file paths should be relative to package directory psf-weather-station
    '''
    def __init__(self, location='cerro-pachon', seed=None,
                 date_range=['2019-05-01', '2019-10-31'],
                 gfs_file='data/gfswinds_cp_20190501-20191031.pkl', 
                 gfs_h_file='data/H.npy',
                 telemetry_file='data/tel_dict_CP_20190501-20191101.pkl'):

        psfws_base = pathlib.Path(__file__).parents[1].absolute()
        self._file_paths = {'gfs_data': pathlib.Path.joinpath(psfws_base, gfs_file),
                            'gfs_alts': pathlib.Path.joinpath(psfws_base, gfs_h_file),
                            'telemetry': pathlib.Path.joinpath(psfws_base, telemetry_file)}
        
        # check that the data files exist:
        for file_path in self._file_paths.values():
            if not file_path.is_file():
                raise FileNotFoundError(f'file {file_path} not found!')

        self.rng = np.random.default_rng(seed)

        # set ground height (location specific) and telescope height
        self.h0, self.h_tel = utils.initialize_location(location)
        
        # load and match GFS/telemetry data
        self._match_data(date_range)

        # set index for lowest GFS data to use according to observatory height:
        # don't use anything lower than 1km above ground
        self.gfs_stop = max([10, np.where(self.h_gfs > self.h0 + 1)[0][0]])

    def _load_data(self):
        '''
        load pickle file of GFS winds and store in class
        '''
        gfs_winds = pickle.load(open(self._file_paths['gfs_data'], 'rb'))
        gfs_winds = utils.process_gfs(gfs_winds)

        # order heights from small to large, and convert to km
        self.h_gfs = np.sort(np.load(open(self._file_paths['gfs_alts'], 'rb'))) / 1000

        telemetry = pickle.load(open(self._file_paths['telemetry'], 'rb'))
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

        matched_s, matched_d, keeper_gfs_dates = utils.match_telemetry(speed, direction, gfs_dates)
        matched_comps = utils.to_components(matched_s, matched_d)
        
        self.telemetry = {'speed': matched_s, 'dir': matched_d, 
                          'u': matched_comps['u'], 'v': matched_comps['v']}
        self.gfs_winds = gfs_winds.loc[keeper_gfs_dates]

        self.N = len(self.gfs_winds)

    def get_raw_wind(self, pt):
        '''
        construct a vector of wind speed+direction vs altitude using GFS,
        till self.gfs_stop, and matched telemetry 
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

    def _draw_ground_cn2(self):
        '''model from "where is surface layer turbulence" Tokovinin paper
        model parameters drawn from normal distribution around paper values'''
        h_gl = np.linspace(self.h_tel,300,100)
        
        tokovinin_a = [-13.38,-13.15,-13.51]
        tokovinin_p = [-1.16,-1.17,-0.97]
            
        a = self.rng.normal(loc=np.mean(tokovinin_a), 
                            scale=np.std(tokovinin_a)/np.sqrt(2))
        p = self.rng.normal(loc=np.mean(tokovinin_p), 
                            scale=np.std(tokovinin_p)/np.sqrt(2))
            
        # return heights of ground layer model, adjusted to observatory height, in km
        return utils.ground_cn2_model({'A':a, 'p':p}, h_gl), self.h0 + h_gl/1000

    def get_cn2(self, pt):
        '''
        use GFS winds and other models to calculate a Cn2 profile
        output is stacked Cn2 of hufnagel model and draw from ground layer model
        '''
        # pick out relevant wind data 
        raw_winds = self.get_raw_wind(pt)

        # make a vector of heights where hufnagel will be valid
        h_huf = np.linspace(self.h0 + 3, max(self.h_gfs), 100) 
        # interpolate wind data to those heights
        speed_huf = self.interpolate_wind(raw_winds, h_huf)['speed'].flatten()
        # calculate hufnagel cn2 for those
        cn2_huf = utils.hufnagel(h_huf, speed_huf)

        # draw model of ground turbulence 
        cn2_gl, h_gl = self._draw_ground_cn2()

        cn2_complete = np.hstack([cn2_gl, cn2_huf])
        h_complete = np.hstack([h_gl, h_huf])

        # return stacked hufnagel and ground layer profiles/altitudes
        return cn2_complete, h_complete

    def get_cn2_all(self):
        '''get array of cn2 values for the whole dataset'''
        cn2_list = []
        for i in range(self.N):
            cn2, h = self.get_cn2(i)
            cn2_list.append(cn2)
        return np.array(cn2_list), h

    def _get_auto_layers(self):
        '''calculate placement of layers in the atmosphere according to the ground, 
        max wind speed, max turbulence, etc'''
        h_interp = np.linspace(self.h0 + self.h_tel, max(self.h_gfs), 500)

        # interpolate the median speeds from GFS to find height of max
        median_spd = np.median([i for i in self.gfs_winds['speed'].values], axis=0)
        median_spd_interp = utils.interpolate(self.h_gfs, median_spd, h_interp, kind='cubic')
        h_max_spd = h_interp[np.argmax(median_spd_interp)]

        # interpolate the median cn2 to find height of max
        all_cn2, h_cn2 = self.get_cn2_all()
        median_cn2 = np.median(all_cn2, axis=0)
        median_cn2_interp = utils.interpolate(h_cn2, median_cn2, h_interp, kind='cubic')
        # find max in mid-atm, not ground
        h_max_cn2 = h_interp[h_interp>2][np.argmax(median_cn2_interp[h_interp>2])] 

        # sort the heights of max speed and max turbulence
        h3, h4 = np.sort([h_max_spd, h_max_cn2])

        # raise the lowest layer slightly off of the ground
        lowest = self.h0 + 0.250

        h2 = np.mean([lowest,h4])
        h5 = np.mean([h4,18])

        return [lowest, h2, h3, h4, h5, 18]

    def _integrate_cn2(self, cn2, h, layers='auto'):
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

        return self._integrate_cn2(cn2, h, layers=layers)

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




