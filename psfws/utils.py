"""Util functions for psfws."""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, make_interp_spline
from scipy.integrate import trapz
import scipy.stats


def convert_to_galsim(params, alt, az, lat=30.2446, lon=70.7494):
    """Convert parameter vector params to coordinates used by GalSim."""
    # rotate wind vector:
    obs_nez, sky_nez = get_both_nez(alt, az, lat, lon)
    
    sky_v, sky_u = [], []
    for v,u in zip(params['v'], params['u']):
        obs_wind = v * obs_nez[0] + u * obs_nez[1]
        sky_wind = np.dot(sky_nez, obs_wind)
        sky_v.append(sky_wind[0])
        sky_u.append(sky_wind[1])

    # modify params: 
    params['v'], params['u'] = sky_v, sky_u
    params['speed'] = np.hypot(sky_v, sky_u)
    params['phi'] = to_direction(sky_v, sky_u)

    # use zenith angle to modify altitudes to LOS distances
    sec_zenith = 1 / np.cos(np.radians(90 - az))
    params['h'] = [h * sec_zenith for h in params['h']]
    params['edges'] = [h * sec_zenith for h in params['edges']]

    # use zenith angle to modify turbulence parameters
    params['j'] = [j * sec_zenith for j in params['j']]
    return params


def get_obs_nez(lat,lon):
    """Get North, East, and zenith unit vectors for observatory at Earth lat,lon."""
    north = np.array([-np.cos(lon)*np.sin(lat),
                      -np.sin(lon)*np.sin(lat),
                      np.cos(lat)])
    zenith = np.array([np.cos(lon)*np.cos(lat),
                       np.sin(lon)*np.cos(lat),
                       np.sin(lat)])
    east = np.cross(north, zenith)
    # unit vectors along _rows_
    return np.array([north, east, zenith])

def get_both_nez(alt, az, lat, lon):
    """Get N, E, zenith unit vectors for observing position alt,az from lat,lon.
    
    Notes:
    - az = 0 means North, 90 means East; alt = 0 means horizon, 90 means zenith
    - Rodrigues' rotation formula 
      (https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula) reduces to: 
          v1 = zenith cos(th) - north sin(th) + 0,
      with v = boresight, k = east, th = -(90 - alt), ie for rot about zenith.
    """
    # convert everything to radians
    lat = -np.deg2rad(lat)
    lon = -np.deg2rad(lon)
    alt = np.deg2rad(alt)
    az = np.deg2rad(az)
    
    # compute n/e/z vectors of observatory
    obs_nez = get_obs_nez(lat,lon)

    # rotate z by zenith then azimuth to get boresight in direction of pointing
    th = -(np.pi/2 - alt)
    v1 = obs_nez[2] * np.cos(th) - obs_nez[0] * np.sin(th)
    boresight = v1 * np.cos(-az) + np.cross(obs_nez[2], v1) * np.sin(-az) \
                + obs_nez[2] * np.dot(obs_nez[2], v1) * (1 - np.cos(-az))

    # compute n/e/z vectors on "the sky"
    lon = np.arctan2(boresight[1], boresight[0])
    lat = np.arcsin(boresight[2])  # assume boresight is normalized
    north = np.array([-np.cos(lon)*np.sin(lat), -np.sin(lon)*np.sin(lat), np.cos(lat)])
    # East = North x boresight
    east = np.cross(north, boresight)
    
    return obs_nez, np.array([north, east, boresight])

def lognorm(sigma, scale):
    """Return a scipy stats lognorm defined by parameters sigma and scale.

    Scale = exp(mu) for mu the mean of the normally distributed variable X such
    that Y=exp(X), and sigma is the standard deviation of X.
    """
    return scipy.stats.lognorm(s=sigma, scale=scale)


def correlate_marginals(X, y, rho, rng):
    """
    Return a joint PDF with specified correlation given two marginal PDFs.

    Parameters
    ----------
    X: pandas dataframe, with wind speed samples in column 'speed'
    y: list or array, samples from second marginal
    rho: float, desired correlation coefficient between X and y
    rng: numpy random state object.

    Returns
    -------
    X: pandas dataframe, same as input X, with added column of y values sorted
       such that the joint PDF has correlation rho.

    """
    X.sort_values(by=['speed'], inplace=True)
    y_srtd = np.sort(y)

    # 15 is ad hoc; seems to work to ensure loops through x at least once
    swp_window = (y_srtd[-1]-y_srtd[0]) / 15
    N = len(y)

    # loop ten times over the dataset
    for i in range(100 * N):
        # index of the first pair in a swap
        i_first = i % N
        # find list of points within the swap_window of this first point
        valid_indices = np.argwhere(abs(y_srtd - y_srtd[i_first]) < swp_window)
        # randomly chose one of these as the swap pair
        i_swp = rng.choice(valid_indices.flatten())
        # swap entries
        y_srtd[i_first], y_srtd[i_swp] = y_srtd[i_swp], y_srtd[i_first]

        if np.corrcoef(X['speed'], y_srtd)[0][1] < rho:
            break

    if np.corrcoef(X['speed'], y_srtd)[0][1] > rho:
        raise ValueError('did not reach desired correlation coefficient!')
    else:
        # add y to the dataframe X as a newcolumn
        try:
            X.insert(loc=2, column='j_gl', value=y_srtd)
        except ValueError:
            print('turbulence column already exists. Check!')

    # return dataframe which now contains the joint PDF
    return X


def process_telemetry(telemetry):
    """Return masked telemetry measurements of speed/direction.

    Input and output are both dicts of pandas series of wind speeds/directions/
    temperatures. Values in output masked for zeros and speeds > 40m/s. Keys in 
    output are 'speed', 'phi', and 't'
    """
    tel_dir = telemetry['wind_direction']
    tel_speed = telemetry['wind_speed']
    tel_temp = telemetry['temperature']

    # find masks for telemetry values that are zero or, for speeds, >40
    speed_mask = tel_speed.index[tel_speed.apply(lambda x: x != 0 and x < 40)]
    dir_mask = tel_dir.index[tel_dir.apply(lambda x: x != 0)]
    temp_mask = tel_temp.index[tel_temp.apply(lambda x: x != 0)]

    # return, converting temperatures to Kelvin from degrees Celsius
    return {'phi': tel_dir.loc[dir_mask],
            'speed': tel_speed.loc[speed_mask],
            't': tel_temp.loc[temp_mask] + 273.15}


def to_direction(x, y):
    """Return wind direction, in degrees, from u,v components of velocity."""
    d = np.arctan2(y, x) * (180/np.pi)
    return (d + 180) % 360


def to_components(s, d):
    """Calculate the wind velocity components given speed/direction."""
    v = s * np.cos((d - 180) * np.pi/180)
    u = s * np.sin((d - 180) * np.pi/180)
    return {'v': v, 'u': u}


def process_forecast(df):
    """Return dataframe of processed global forecasting system data.

    Input: dataframe of forecast obsevrations, cols = ['u', 'v', 't']
    Returns: dataframe of forecast observations,
             cols = ['u', 'v', 't', 'speed', 'phi']

    Processing steps:
    - reverse u, v, and t altitudes
    - filter daytime datapoints
    - add "speed" and "phi" columns
    """
    not_daytime = df.index.hour != 12
    df = df.iloc[not_daytime].copy()
    n = len(df)

    # reverse
    for k in ['u', 'v', 't']:
        df[k] = [df[k].values[i][::-1] for i in range(n)]

    df['speed'] = [np.hypot(df['u'].values[i], df['v'].values[i])
                   for i in range(n)]

    df['phi'] = [to_direction(df['v'].values[i], df['u'].values[i])
                 for i in range(n)]

    if 'p' in df.columns:
        df.drop('p', axis=1, inplace=True)

    return df


def match_telemetry(telemetry, forecast_dates):
    """Return overlap between telemetry and forecasting data.

    Parameters
    ----------
    telemetry: pandas dict
        Columns: 'speed', 'dir', 't', and index of datetimes of observations.

    forecast_dates: pd Series containing datetime objects of forecasts

    Returns
    -------
    matched telemetry: dict
        Parameters returned are medians in 1h bins around each forecast. Dict
        has keys 'speed', 'dir', 't', 'u', 'v'.
    dates: pd Series
        subselection of input dates which had a valid overlap.

    """
    n = len(forecast_dates)

    speed = telemetry['speed']
    direction = telemetry['phi']
    temp = telemetry['t']

    speed_ids = [speed.index[abs(forecast_dates[i] - speed.index)
                             < pd.Timedelta('30min')] for i in range(n)]

    dir_ids = [direction.index[abs(forecast_dates[i] - direction.index)
                               < pd.Timedelta('30min')] for i in range(n)]

    temp_ids = [temp.index[abs(forecast_dates[i] - temp.index)
                           < pd.Timedelta('30min')] for i in range(n)]

    ids_to_keep = [i for i in range(n)
                   if len(speed_ids[i]) != 0
                   and len(dir_ids[i]) != 0
                   and len(temp_ids[i]) != 0]

    matched_s = [np.median(speed.loc[speed_ids[i]])
                 for i in range(n) if i in ids_to_keep]
    matched_d = [np.median(direction.loc[dir_ids[i]])
                 for i in range(n) if i in ids_to_keep]
    matched_t = [np.median(temp.loc[temp_ids[i]])
                 for i in range(n) if i in ids_to_keep]

    # calculate velocity componenents from the matched speed/directions
    uv = to_components(np.array(matched_s), np.array(matched_d))

    return ({'speed': np.array(matched_s), 'phi': np.array(matched_d),
             't': np.array(matched_t), 'u': uv['u'], 'v': uv['v']},
            forecast_dates[ids_to_keep])


def interpolate(x, y, new_x, ddz=True, s=0):
    """Interpolate 1D array y at values x to new_x values.

    Parameters
    ----------
    x : array
        x values of input y
    y : array
        1D array to interpolate
    new_x : array
        x values of desired interpolated output
    ddz : bool (default True)
        Whether or not to return the derivative of y wrt z
    s : float or None (Default)
        Smoothing factor used to choose number of knots. Use None to let
        scipy use their best estimate. Use s=0 for perfect interpolation.

    Returns
    -------
        new_y, interpolated values of y at positions new_x
        dydz, derivative of new_y at positions new_x, if ddz=True

    """
    # this is a smoothing spline unless s=0
    f_y = UnivariateSpline(x, y, s=s)

    if ddz:
        dfydz = f_y.derivative()
        return f_y(new_x), dfydz(new_x)
    else:
        return f_y(new_x)


def osborn(inputs):
    """Calculate Cn2 model from Osborn et al 2018."""
    g = 9.8

    # theta and d/dz(theta)
    thetaz, dthetaz = osborn_theta(inputs)

    # wind shear => caclulate L(h)
    windshear = inputs['dudz']**2 + inputs['dvdz']**2
    lz = np.sqrt(2*thetaz / g * windshear / abs(dthetaz))

    numerator = 80e-6 * inputs['p'] * dthetaz
    denominator = inputs['t'] * thetaz

    return lz**(4/3) * (numerator / denominator)**2


def osborn_theta(inputs):
    """Calculate potential temperature theta and its derivative."""
    Rcp = 0.286
    P0 = 1000 * 100  # mbar to Pa

    theta = inputs['t'] * (P0 / inputs['p'])**Rcp

    amp = (P0/inputs['p'])**Rcp
    p_ratio = inputs['dpdz'] / inputs['p']
    dthetaz = amp * (inputs['dtdz'] - Rcp * inputs['t'] * p_ratio)

    return theta, dthetaz


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
        # get Cn2 interpolation at those values of h -- s=0 for no smoothing.
        cn2_i = np.exp(interpolate(h * 1000, np.log(cn2), h_i, ddz=False, s=0))
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
