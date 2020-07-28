import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

def process_telemetry(telemetry_df):
    '''
    input: telemetry object
    returns 
    - a dict holding a dataframe each for wind directions and speeds
    - a dict with masks for each of the above dfs
    '''
    tel_dir = pd.DataFrame(telemetry_df['WindDir_twr']) 
    tel_speed = pd.DataFrame(telemetry_df['WindSpd_twr'])

    for dframe in [tel_dir, tel_speed]:
        dframe.index = pd.to_datetime(dframe['dts'], utc=True)

    nonzero = tel_speed.index[tel_speed['vals'] != 0]
    cut = tel_speed.index[tel_speed['vals'] < 40]
    cp_masks = {'speed': nonzero & cut, 'dir': tel_dir.sample(n=50_000).index}

    return {'dir': tel_dir, 'speed': tel_speed}, cp_masks

def process_gfs(gfs_df):
    '''
    input: dataframe of GFS obsevrations with 'u' and 'v' columns
    returns dataframe with: 
    - u and v altitudes reversed
    - new "speed" and "dir" columns added
    '''
    gfs_df['u'] = [gfs_df['u'].values[i][::-1] for i in range(len(gfs_df))]
    gfs_df['v'] = [gfs_df['v'].values[i][::-1] for i in range(len(gfs_df))]

    gfs_df['speed'] = [np.hypot(gfs_df['u'].values[i], gfs_df['v'].values[i])
                       for i in range(len(gfs_df))]

    gfs_df['dir'] = [smooth_direction(gfs_df['v'].values[i], gfs_df['u'].values[i]) 
                     for i in range(len(gfs_df))]
    return gfs_df

def match_telemetry(speed, direction, gfs_dates):
    '''
    return matched speed/direction data associated with the gfs dates 
    specifically, return the median telemetry values in a +/- 30min interval around each GFS point
    '''
    speed_ids = [speed.index[abs(gfs_dates[i] - speed.index) < pd.Timedelta('30min')] 
               for i in range(n_gfs)]
    dir_ids = [direction.index[abs(gfs_dates[i] - direction.index) < pd.Timedelta('30min')] 
               for i in range(n_gfs)]
    
    ids_to_keep = [i for i in range(n_gfs) if len(speed_ids[i])!=0 and len(dir_ids)!=0]
    
    matched_s = [np.median(speed.loc[speed_ids[i]]['vals']) for i in range(n_gfs) if i in ids_to_keep]
    matched_d = [np.median(direction.loc[dir_ids[i]]['vals']) for i in range(n_gfs) if i in ids_to_keep]

    return matched_s, matched_d

def interpolate(x, y):
    new_x = np.linspace(min(x), max(x), 100)
    f_y = interp1d(x, y, kind='cubic')
    new_y = f_y(new_x)
    return new_y

def smooth_direction(V,U):
    '''
    Convert U, V components of wind velocity to a direction.
    Returns the smoothest answer (i.e. shifts points +/- 360 for a smooth curve)
    '''
    wind_dir = np.arctan2(U, V)*(180/np.pi) 
    
    smooth_dir = np.zeros(wind_dir.shape)
    smooth_dir[0] = wind_dir[0]
    
    for i in range(smooth_dir.shape[0]-1):
        # make an array of options: next pt, next pt+360, next pt-360
        options = wind_dir[i+1] + np.array([720, 360, 0, -360, -720])
        # find the option for next that has smallest jump from the current
        smoothest = np.argmin(abs(options - smooth_dir[i]))
        smooth_dir[i+1] = options[smoothest]

    # check the mean of the directions, bring up/down by 360 if needed
    while smooth_dir.mean() > 180:
        smooth_dir -= 360
    while smooth_dir.mean() < -180: 
        smooth_dir += 360
        
    return smooth_dir