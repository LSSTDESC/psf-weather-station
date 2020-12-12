"""Util functions for psfws."""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapz


def initialize_location(location):
    """Given location identifier, return ground altitude and dome height in km.

    Parameters
    ----------
    location : str or dict
        Valid options for : 'cerro-paranal', 'cerro-pachon', 'cerro-telolo',
        'mauna-kea', and 'la-palma'.
        To customize to another observatory, input instead a dict with keys
        'altitude' (value in km) and 'height' (optional: dome height in m.
        If not given, will default to 10m)

    """
    # custom usage
    if type(location) == dict:
        h0 = location['altitude']
        try:
            h_tel = location['height']
        except KeyError:
            h_tel = 10

    else:
        ground_altitude = {'cerro-pachon': 2.715, 'mauna-kea': 4.2,
                           'cerro-telolo': 2.2, 'la-palma': 2.4,
                           'cerro-paranal': 2.64}
        h0 = ground_altitude[location]
        h_tel = 10  # in m

    return h0, h_tel/1000


def ground_cn2_model(params, h):
    """Return power law model for Cn2 as function of elevation."""
    logcn2 = params['A'] + params['p'] * np.log10(h)
    return 10**(logcn2)


def find_max_median(x, h_old, h_new, h0):
    """Find max of median of array x by interpolating datapoints."""
    # interpolate median x to smoothly spaced new h values
    x_median = np.median(x, axis=0)
    x_interp = interpolate(h_old, x_median, h_new, kind='cubic')

    # find maximum of interpolated x *above 2km*, to avoid ground effects
    max_index = np.argmax(x_interp[h_new > 2 + h0])
    h_max = h_new[h_new > 2 + h0][max_index]

    return h_max


def process_telemetry(telemetry):
    """Return masked telemetry measurements of speed/direction.

    Input and output are both dicts of pandas series of wind speeds/directions.
    Values in output masked for zeros and speeds > 40m/s. Keys in output are
    'speed' and 'dir'
    """
    tel_dir = telemetry['wind_direction']
    tel_speed = telemetry['wind_speed']

    # find masks for telemetry values that are zero or, for speeds, >40
    speed_mask = tel_speed.index[tel_speed.apply(lambda x: x != 0 and x < 40)]
    dir_mask = tel_dir.index[tel_dir.apply(lambda x: x != 0)]

    return {'dir': tel_dir.loc[dir_mask], 'speed': tel_speed.loc[speed_mask]}


def to_direction(x, y):
    """Return wind direction, in degrees, from u,v components of velocity."""
    d = np.arctan2(y, x) * (180/np.pi)
    return (d + 180) % 360


def to_components(s, d):
    """Calculate the wind velocity components given speed/direction."""
    v = s * np.cos((d - 180) * np.pi/180)
    u = s * np.sin((d - 180) * np.pi/180)
    return {'v': v, 'u': u}


def process_gfs(gfs_df):
    """Return dataframe of processed global forecasting system data.

    Input: dataframe of GFS obsevrations, columns = ['u', 'v']
    Returns: dataframe of GFS observations, columns = ['u','v','speed','dir']

    Processing steps:
    - reverse u and v altitudes
    - filter daytime datapoints
    - add "speed" and "dir" columns
    """
    not_daytime = gfs_df.index.hour != 12
    gfs_df = gfs_df.iloc[not_daytime].copy()
    n = len(gfs_df)

    # reverse, and disregard the top 5 altitudes
    gfs_df['u'] = [gfs_df['u'].values[i][::-1][:-5] for i in range(n)]
    gfs_df['v'] = [gfs_df['v'].values[i][::-1][:-5] for i in range(n)]

    gfs_df['speed'] = [np.hypot(gfs_df['u'].values[i], gfs_df['v'].values[i])
                       for i in range(n)]

    gfs_df['dir'] = [to_direction(gfs_df['v'].values[i], gfs_df['u'].values[i])
                     for i in range(n)]
    return gfs_df


def match_telemetry(telemetry, gfs_dates):
    """Return speed, dir, and overlap datetimes between telemetry and GFS.

    Telemetry vals returned are medians in 1h bins around each GFS output.
    """
    n_gfs = len(gfs_dates)

    speed = telemetry['speed']
    direction = telemetry['dir']

    speed_ids = [speed.index[abs(gfs_dates[i] - speed.index)
                             < pd.Timedelta('30min')] for i in range(n_gfs)]

    dir_ids = [direction.index[abs(gfs_dates[i] - direction.index)
                               < pd.Timedelta('30min')] for i in range(n_gfs)]

    ids_to_keep = [i for i in range(n_gfs)
                   if len(speed_ids[i]) != 0 and len(dir_ids[i]) != 0]

    matched_s = [np.median(speed.loc[speed_ids[i]])
                 for i in range(n_gfs) if i in ids_to_keep]
    matched_d = [np.median(direction.loc[dir_ids[i]])
                 for i in range(n_gfs) if i in ids_to_keep]

    return np.array(matched_s), np.array(matched_d), gfs_dates[ids_to_keep]


def interpolate(x, y, new_x, kind):
    """Interpolate 1D array y at values x to new_x values.

    Parameters
    ----------
    x : array
        x values of input y
    y : array
        1D array to interpolate
    new_x : array
        x values of desired interpolated output
    kind : str
        Perform either 'cubic' or 'gp' interpolation

    Returns
    -------
        new_y, interpolated values of y at positions new_x

    """
    if kind == 'cubic':
        f_y = interp1d(x, y, kind='cubic')
        new_y = f_y(new_x)
    elif kind == 'gp':
        from sklearn import gaussian_process
        gp = gaussian_process.GaussianProcessRegressor(normalize_y=True,
                                                       alpha=1e-3,
                                                       n_restarts_optimizer=5)
        gp.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        new_y = gp.predict(new_x.reshape(-1, 1))
    return new_y


def hufnagel(z, v):
    """Calculte Hufnagel Cn2, input z in km, v in m/s."""
    if np.min(z) < 3:
        raise ValueError('hufnagel model only valid for height>3km')
    z = np.copy(z) * 1000  # to km
    tmp1 = 2.2e-53 * z**10 * (v / 27)**2 * np.exp(-z * 1e-3)
    tmp2 = 1e-16 * np.exp(-z * 1.5e-3)
    return tmp1 + tmp2


def integrate_in_bins(cn2, h, edges):
    """Integrate cn2 into altitude bins.

    Parameters:
    -----------
    cn2 : array, values of cn2
    h : array, altitudes of input cn2 values
    edges: array, edges of integration regions

    Returns:
    -------
    turbulence integral J : array, size len(edges)-1

    """
    # make an equally spaced sampling in h across whole range:
    h_samples = np.linspace(edges[0] * 1000, edges[-1] * 1000, 1000)

    J = []
    for i in range(len(edges)-1):
        # find the h samples that are within the altitude range of integration
        h_i = h_samples[(h_samples < edges[i+1] * 1000) &
                        (h_samples > edges[i] * 1000)]
        # get Cn2 interpolation at those values of h
        cn2_i = np.exp(interpolate(h * 1000, np.log(cn2), h_i, kind='cubic'))
        # numerically integrate to find the J value for this bin
        J.append(trapz(cn2_i, h_i))

    return J


def smooth_direction(directions):
    """Return "smoothed" dirs by shifting points +/- 360 for smooth curve."""
    smooth_dir = np.zeros(directions.shape)
    smooth_dir[0] = directions[0]

    for i in range(smooth_dir.shape[0]-1):
        # make an array of options: next pt, next pt+360, next pt-360
        options = directions[i+1] + np.array([720, 360, 0, -360, -720])
        # find the option for next that has smallest jump from the current
        smoothest = np.argmin(abs(options - smooth_dir[i]))
        smooth_dir[i+1] = options[smoothest]

    # check the mean of the directions, bring up/down by 360 if needed
    while smooth_dir.mean() > 180:
        smooth_dir -= 360
    while smooth_dir.mean() < -180:
        smooth_dir += 360

    return smooth_dir
