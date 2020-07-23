import numpy as np
import pandas as pd

def process_telemetry(telemetry_df):
    '''
    input: telemetry object
    returns 
    - a dict holding a dataframe each for wind directions and speeds
    - a dict with masks for each of the above dfs
    '''
    cp_wind_dir = pd.DataFrame(telemetry_df['WindDir_twr']) 
    cp_wind_speed = pd.DataFrame(telemetry_df['WindSpd_twr'])

    for dframe in [cp_wind_dir, cp_wind_speed]:
        dframe.index = pd.to_datetime(dframe['dts'], utc=True)

    cp_masks = {'speed': cp_wind_speed != 0}

    return {'dir': cp_wind_dir, 'speed': cp_wind_speed}, cp_masks


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