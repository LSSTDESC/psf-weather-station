"""Util functions for psfws."""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, make_interp_spline
from scipy.integrate import trapz
from csaps import CubicSmoothingSpline
import scipy.stats


def lognorm(sigma, scale):
    """Return a scipy stats lognorm defined by parameters sigma and scale.

    Scale = exp(mu) for mu the mean of the normally distributed variable X such
    that Y=exp(X), and sigma is the standard deviation of X."""
    return scipy.stats.lognorm(s=sigma, scale=scale)

def correlate_marginals(X, y, rho):
    """
    Takes two marginal distrubtions, x and y, and returns a joint PDF with
    specified correlation coefficient rho. Also returns resulting correlation.

    Parameters
    ----------
    x: pandas dataframe, wind speed values 
    y: list or array, ground turbulence integrals 
    rho: float, desired correlation coefficient between x and y

    """
    X.sort_values(by=['speed'], inplace=True)
    y_srtd = np.sort(y)

    # 15 is ad hoc; seems to work to ensure loops through x at least once
    swap_window = (y_srtd[-1]-y_srtd[0]) / 15
    N = len(y)

    # loop ten times over the dataset
    for i in range(10 * N):
        # index of the first pair in a swap
        i_first = i % N
        # find list of points within the swap_window of this first point
        valid_indices = np.argwhere(abs(y_srtd - y_srtd[i_first]) < swap_window)
        # randomly chose one of these as the swap pair
        i_swap = np.random.choice(valid_indices)
        # swap entries
        y_srtd[i_first], y_srtd[i_swap] = y_srtd[i_swap], y_srtd[i_first]

        if np.corrcoeff(X['speed'], y_srtd)[0][1] < rho:
            break

    if np.corrcoeff(X['speed'], y_srtd)[0][1] > rho:
        raise ValueError('did not reach desired correlation coefficient!')
    else:
        # add y to the dataframe X as a newcolumn
        try: 
            X.insert(loc=2, col='j_gl', values=y_srtd)
        except ValueError:
            print('turbulence column already exists. Check!')

    # return dataframe which now contains the joint PDF 
    return X

def initialize_location(loc):
    """Given location identifier, return ground altitude and dome height in km.

    Parameters
    ----------
    loc : str or dict
        Valid options for : 'cerro-paranal', 'cerro-pachon', 'cerro-telolo',
        'mauna-kea', and 'la-palma'.
        To customize to another observatory, input instead a dict with keys
        'altitude' (value in km) and 'turbulence_params'

    """
    # custom usage
    if type(loc) == dict:
        h0 = loc['altitude']
        j_params = loc['turbulence_params']

    elif type(loc) == str:
        ground_altitude = {'cerro-pachon': 2.715, 
                           'mauna-kea': 4.2,
                           'cerro-telolo': 2.2, 
                           'la-palma': 2.4,
                           'cerro-paranal': 2.64}
        j_params = {'cerro-pachon': {'gl': {'s': 0.62, 'scale': 2.34},
                                           'fa': {'s': 0.84, 'scale': 1.51}}}

    else:
        return TypeError('loc arg must be either dict or string!')
 
    # initialize lognorm pdfs for ground and FA turbulence:
    j_pdf = {k: lognorm(v['s'], v['scale']) for k,v in j_params[loc].items()}

    try:
        return ground_altitude[loc], j_pdf
    except KeyError:
        print('loc must be one of allowed locations! See docstring.')


def ground_cn2_model(params, h):
    """Return power law model for Cn2 as function of elevation."""
    logcn2 = params['A'] + params['p'] * np.log10(h)
    return 10**(logcn2)


def find_max_median(x, h_old, h_new, h0):
    """Find max of median of array x by interpolating datapoints."""
    # interpolate median x to smoothly spaced new h values
    if len(np.array(x).shape) >= 2:
        x = np.median(x, axis=0)

    x_interp = interpolate(h_old, x, h_new, ddz=False)

    # find maximum of interpolated x *above 2km*, to avoid ground effects
    max_index = np.argmax(x_interp[h_new > 2 + h0])
    h_max = h_new[h_new > 2 + h0][max_index]

    return h_max


def process_telemetry(telemetry):
    """Return masked telemetry measurements of speed/direction.

    Input and output are both dicts of pandas series of wind speeds/directions/
    temperatures.
    Values in output masked for zeros and speeds > 40m/s. Keys in output are
    'speed', 'dir', and 'temp'
    """
    tel_dir = telemetry['wind_direction']
    tel_speed = telemetry['wind_speed']
    tel_temp = telemetry['temperature']

    # find masks for telemetry values that are zero or, for speeds, >40
    speed_mask = tel_speed.index[tel_speed.apply(lambda x: x != 0 and x < 40)]
    dir_mask = tel_dir.index[tel_dir.apply(lambda x: x != 0)]
    temp_mask = tel_temp.index[tel_temp.apply(lambda x: x != 0)]

    # return, converting temperatures to Kelvin from degrees Celsius
    return {'dir': tel_dir.loc[dir_mask],
            'speed': tel_speed.loc[speed_mask],
            'temp': tel_temp.loc[temp_mask] + 273.15}


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

    Input: dataframe of GFS obsevrations, cols = ['u', 'v', 't', 'p']
    Returns: dataframe of GFS observations,
             cols = ['u', 'v', 't', 'p', 'speed', 'dir']

    Processing steps:
    - reverse u, v, and t altitudes
    - filter daytime datapoints
    - add "speed" and "dir" columns
    """
    not_daytime = gfs_df.index.hour != 12
    gfs_df = gfs_df.iloc[not_daytime].copy()
    n = len(gfs_df)

    # reverse, and disregard the top 5 altitudes
    for k in ['u', 'v', 't', 'p']:
        gfs_df[k] = [gfs_df[k].values[i][::-1][:-5] for i in range(n)]

    gfs_df['speed'] = [np.hypot(gfs_df['u'].values[i], gfs_df['v'].values[i])
                       for i in range(n)]

    gfs_df['dir'] = [to_direction(gfs_df['v'].values[i], gfs_df['u'].values[i])
                     for i in range(n)]

    # find altitudes of GFS outputs; all data have same pressures, so take first
    median_t = np.median([gfs_df['t'].values[i] for i in range(n)], axis=0)
    h_gfs = pressure_to_h(gfs_df['p'].values[0], median_t)
    # gfs_df['h'] = [pressure_to_h(gfs_df['p'].values[0], 
    #                              gfs_df['t'].values[i]) for i in range(n)]
    return gfs_df, h_gfs


def pressure_to_h(p, t):
    """Convert array of pressure and temperature values to altitude."""
    M = 28.96  # average mol. mass of atmosphere, in g/mol
    g = 9.8  # accelerationg at the surface
    R = 8.314  # gas constant, in J/mol/K
    P0 = 1013  # pressure at sea level, in hPa (=mbar)

    return (R * t) / (M * g) * np.log(P0 / p)


def match_telemetry(telemetry, gfs_dates):
    """Return speed, dir, and overlap datetimes between telemetry and GFS.

    Telemetry vals returned are medians in 1h bins around each GFS output.
    """
    n_gfs = len(gfs_dates)

    speed = telemetry['speed']
    direction = telemetry['dir']
    temp = telemetry['temp']

    speed_ids = [speed.index[abs(gfs_dates[i] - speed.index)
                             < pd.Timedelta('30min')] for i in range(n_gfs)]

    dir_ids = [direction.index[abs(gfs_dates[i] - direction.index)
                               < pd.Timedelta('30min')] for i in range(n_gfs)]

    temp_ids = [temp.index[abs(gfs_dates[i] - temp.index)
                           < pd.Timedelta('30min')] for i in range(n_gfs)]

    ids_to_keep = [i for i in range(n_gfs)
                   if len(speed_ids[i]) != 0
                   and len(dir_ids[i]) != 0
                   and len(temp_ids[i]) != 0]

    matched_s = [np.median(speed.loc[speed_ids[i]])
                 for i in range(n_gfs) if i in ids_to_keep]
    matched_d = [np.median(direction.loc[dir_ids[i]])
                 for i in range(n_gfs) if i in ids_to_keep]
    matched_t = [np.median(temp.loc[temp_ids[i]])
                 for i in range(n_gfs) if i in ids_to_keep]

    return (np.array(matched_s), np.array(matched_d),
            np.array(matched_t), gfs_dates[ids_to_keep])


def interpolate(x, y, new_x, ddz=True, extend=True):
    """Interpolate 1D array y at values x to new_x values.

    Parameters
    ----------
    x : array
        x values of input y
    y : array
        1D array to interpolate
    new_x : array
        x values of desired interpolated output
    ddz : bool
        Whether or not to return the derivative of y wrt z

    Returns
    -------
        new_y, interpolated values of y at positions new_x
        dydz, derivative of new_y at positions new_x, if ddz=True

    """
    if extend:
        # make a spline and fix constant first derivative at the boundary
        # spline = make_interp_spline(x, y, bc_type='natural')
        # # get the spline extrapolation above/below data region (ok bc f'=const)
        # delta_x = x[1]-x[0]
        # x_below = np.linspace(x[0]-2*delta_x, x[0], 20)
        # x_above = np.linspace(x[-1], x[-1]+2*delta_x, 20)
        # y = np.concatenate([spline(x_below), y, spline(x_above)])
        # x = np.concatenate([x_below, x, x_above])
        s = CubicSmoothingSpline(x, y, smooth=None).spline
        if ddz: 
            return s(new_x), s.derivative(nu=1)(new_x)
        else: 
            return s(new_x)
    else:
        # now use a smoothing spline
        f_y = UnivariateSpline(x, y, s=s)

        if ddz:
            dfydz = f_y.derivative()
            return f_y(new_x), dfydz(new_x)
        else:
            return f_y(new_x)


def osborn(inputs, k=1):
    """Calculate Cn2 model from Osborn et al 2018."""
    g = 9.8

    # theta and d/dz(theta)
    thetaz, dthetaz = osborn_theta(inputs)

    # wind shear => caclulate L(h)
    windshear = inputs['dudz']**2 + inputs['dvdz']**2
    lz = np.sqrt(2*thetaz / g * windshear / abs(dthetaz))

    numerator = 80e-6 * inputs['p'] * dthetaz
    denominator = inputs['t'] * thetaz

    return k * lz**(4/3) * (numerator / denominator)**2


def osborn_theta(inputs):
    """Calculate potential temperature theta and its derivative."""
    Rcp = 0.286
    P0 = 1000 * 100  # mbar to Pa

    theta = inputs['t'] * (P0 / inputs['p'])**Rcp

    amp = (P0/inputs['p'])**Rcp
    p_ratio = inputs['dpdz'] / inputs['p']
    dthetaz = amp * (inputs['dtdz'] - Rcp * inputs['t'] * p_ratio)

    return theta, dthetaz


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
        cn2_i = np.exp(interpolate(h * 1000, np.log(cn2), h_i, ddz=False))
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
