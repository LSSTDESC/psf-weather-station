"""Functions to download and reformat ECMWF forecast data."""

import numpy as np
import pandas as pd
import pickle
import pathlib
import os
import compute_geopotential_on_ml as ecmwfUtil

PKG_BASE = pathlib.Path(__file__).resolve().parents[0].absolute()
PKG_DATA_DIR = pathlib.Path.joinpath(PKG_BASE, 'data/')


def get_geometric_h(geopotential):
    """Convert geopotential to geometric height."""
    Rearth = 6.371e6  # in meters
    g = 9.80665  # m/s^2
    # geopotential height is relative to the geoid, where geoid is shape of
    # spherical Earth due to gravity/etc
    geopotential_h = geopotential / g
    # geometric height is also relative to the geoid
    geometric_h = Rearth * geopotential_h / (Rearth - geopotential_h)
    return geometric_h


def get_geopotential(idx, step, values):
    """Compute z at half & full level for the given level, based on t/q/sp."""
    # We want to integrate up into the atmosphere, starting at the
    # ground so we start at the lowest level (highest number) and
    # keep accumulating the height as we go.
    # See the IFS documentation, part III
    # For speed and file I/O, we perform the computations with
    # numpy vectors instead of fieldsets.
    # Modified from ecmwfUtil.production_step()
    z_h = values['z']
    hout = []
    pout = []
    for lev in sorted(values['levelist'], reverse=True):
        try:
            z_h, z_f, p = ecmwfUtil.compute_z_level(idx, lev, values, z_h)
            hout.append(float(z_f))
            pout.append(float(p))
        except MissingLevelError as e:
            print('%s [WARN] %s' % (sys.argv[0], e), file=sys.stderr)
    return np.stack((get_geometric_h(np.array(hout)), np.array(pout)))


def _height_download(date, lat, lon):
    import cdsapi
    cds = cdsapi.Client()
    cds.retrieve('reanalysis-era5-complete', {
                 'class': 'ea',
                 'expver': '1',
                 'stream': 'oper',
                 'time': '04',  # this would have to be changed!
                 'type': 'an',
                 'grid': '0.25/0.25',
                 'area': f"{lat}/{lon}/{lat}/{lon}",
                 'date': date,
                 'levelist': '1',  # Geopotential (z) and Logarithm of surface pressure (lnsp) are 2D fields, archived as model level 1
                 'levtype': 'ml',
                 'param': '129/152',  # Geopotential (z) and Logarithm of surface pressure (lnsp)
                 }, 'zlnsp_ml.grib')

    cds.retrieve('reanalysis-era5-complete', {
                  'class': 'ea',
                  'date': date,
                  'area': f"{lat}/{lon}/{lat}/{lon}",  # N/W/S/E bounds
                  'expver': '1',
                  'levelist': '37/to/137',  # don't need high altitude levels
                  'levtype': 'ml',  # model level outputs (finely sampled in h)
                  'param': '130/133',  # codes for temp, specific humidity
                  'stream': 'oper',
                  'time': '04',  # output times to download # this would have to be changed!
                  'type': 'an',  # 'an' for analysis
                  'grid': '0.25/0.25',  # need this to get lat/lon outputs!!
                  'format': 'grib',  # output file format
                  }, 'tq_ml.grib')


def geopotential_on_ml(date, lat, lon, grib_dir, levelist='37/to/137', output='ph_{}_{}.csv'):
    """Download data and calculate geometric height array for this location."""
    # Main function from ecmwfUtil.main()
    import eccodes
    date = date[:4]+'-'+date[4:6]+'-'+date[6:]

    zsp_path = pathlib.Path.joinpath(grib_dir, 'zlnsp_ml.grib')
    tq_path = pathlib.Path.joinpath(grib_dir, 'tq_ml.grib')
    if not zsp_path.exists() or not tq_path.exists():
        _height_download(date, lat, lon)

    if levelist == 'all':
        levelist = range(1, 138)
    else:
        levels = levelist.split('/')
        levelist = list(range(int(levels[0]), int(levels[2]) + 1))

    index_keys = ['date', 'time', 'shortName', 'level', 'step']

    idx = eccodes.codes_index_new_from_file('zlnsp_ml.grib', index_keys)
    eccodes.codes_index_add_file(idx, 'tq_ml.grib')
    values = None

    # iterate date
    for date in eccodes.codes_index_get(idx, 'date'):
        eccodes.codes_index_select(idx, 'date', date)
        # iterate time
        for time in eccodes.codes_index_get(idx, 'time'):
            eccodes.codes_index_select(idx, 'time', time)
            for step in eccodes.codes_index_get(idx, 'step'):
                eccodes.codes_index_select(idx, 'step', step)
                if not values:
                    values = ecmwfUtil.get_initial_values(idx, keep_sample=True)
                values['levelist'] = levelist
                values['sp'] = ecmwfUtil.get_surface_pressure(idx)
                out = get_geopotential(idx, step, values)
                break
            break
        break  # break statements because we only need z for one date/time
    np.savetxt(output.format(lat, lon), out)


def _download_ecmwf(times, m1, m2, lat, lon, save_path):
    """Download ECMWF model level data.

    Parameters
    ==========
    m1 : str
        starting date of data to collect, formatted as "YYYY-MM-DD".
    m2 : str
        ending date of data to collect, formatted as "YYYY-MM-DD".
    lat : float
        latitude of site of interest, to nearest 0.25 degrees
    lon : float
        longitude of site of interest, to nearest 0.25 degrees
    save_path : str
        path to file to save the downloaded info, should be in .grib format.

    Notes:
    - best practice is to query month by month, so this function is inteded to
      be called >1 times if the desired date range spans multiple months.
    """
    import cdsapi
    cds = cdsapi.Client()

    # check that the lat/lon values are %.25 = 0
    lat_lon_check = lat % 0.25 + lon % 0.25
    assert lat_lon_check == 0, "Lat, lon must be rounded to nearest 0.25 deg!"

    cds.retrieve('reanalysis-era5-complete', {
                  'class': 'ea',
                  'date': f"{m1}/to/{m2}",
                  'area': f"{lat}/{lon}/{lat}/{lon}",  # N/W/S/E bounds
                  'expver': '1',
                  'levelist': '37/to/137',  # don't need high altitude levels
                  'levtype': 'ml',  # model level outputs (finely sampled in h)
                  'param': '130/131/132'+p,  # codes for temp, wind u/v
                  'stream': 'oper',
                  'time': times,  # output times to download # this would have to be changed!
                  'type': 'an',  # 'an' for analysis
                  'grid': '0.25/0.25',  # need this to get lat/lon outputs!!
                  'format': 'grib',  # output file format
                  }, save_path)
    return


def _get_month_edges(date):
    """Return Timestamps for start and end of the month of given date."""
    month_start = pd.Timestamp(year=date.year, month=date.month, day=1)
    if date.month == 12:
        month_end = pd.Timestamp(year=date.year, month=date.month, day=31)
    else:
        next_month = pd.Timestamp(year=date.year, month=date.month+1, day=1)
        month_end = next_month - pd.Timedelta(days=1)
    return month_start, month_end


def _get_iter_months(start_date, end_date):
    """Return list of date pairs to iterate through months of interest."""
    d1 = pd.Timestamp(start_date)
    d2 = pd.Timestamp(end_date)

    # years are the same for this function
    if d1.month == d2.month:
        dates = [(d1, d2)]
    if d1.month == d2.month - 1:
        dates = [(d1, _get_month_edges(d1)[1]), (_get_month_edges(d2)[0], d2)]
    else:
        middle_months = np.arange(d1.month + 1, d2.month)
        # day doesn't matter but needed as argument to Timestamp
        middle_pairs = [_get_month_edges(pd.Timestamp(year=d1.year,
                                                      month=m, day=1))
                        for m in middle_months]

        dates = [(d1, _get_month_edges(d1)[1])] + middle_pairs + \
                [(_get_month_edges(d2)[0], d2)]

    return dates


def _get_iter_dates(start_date, end_date):
    """Return list of date pairs in month increments over dates of interest."""
    d1 = pd.Timestamp(start_date)
    d2 = pd.Timestamp(end_date)

    # if years are the same:
    if d1.year == d2.year:
        dates = _get_iter_months(d1, d2)
    # if years are adjacent, only have the edge cases:
    elif d1.year == d2.year - 1:
        y1_end = pd.Timestamp(year=d1.year, month=12, day=31)
        y2_start = pd.Timestamp(year=d2.year, month=1, day=1)
        dates = _get_iter_months(d1, y1_end) + _get_iter_months(y2_start, d2)
    # otherwise, iterate through years and get the months from _get_iter_months
    else:
        # iterate through the years in between d1 and d2 (inclusive) and
        # return all the resulting iter_month results.
        y1_end = pd.Timestamp(year=d1.year, month=12, day=31)
        dates = [_get_iter_months(d1, y1_end)]

        for y in np.arange(d1.year + 1, d2.year):
            y_start = pd.Timestamp(year=y, month=1, day=1)
            y_end = pd.Timestamp(year=y, month=12, day=31)
            dates += _get_iter_months(y_start, y_end)

        y2_start = pd.Timestamp(year=d2.year, month=1, day=1)
        dates += _get_iter_months(y2_start, d2)

    return dates


def get_values(idx, step, levelist):
    """Extract values from level."""
    import eccodes
    t, u, v = [], [], []
    for lev in sorted(levelist, reverse=True):
        eccodes.codes_index_select(idx, 'level', lev)
        for short, outlist in zip(['t', 'u', 'v'], [t, u, v]):
            try:
                eccodes.codes_index_select(idx, 'shortName', short)
                gid = eccodes.codes_new_from_index(idx)
                outlist.append(float(eccodes.codes_get_values(gid)))
                eccodes.codes_release(gid)
            except MissingLevelError as e:
                print('%s [WARN] %s' % (sys.argv[0], e), file=sys.stderr)
    return {'t': t, 'u': u, 'v': v}


def _process_grib(infile):
    """Open downloaded grib file, add data to t,u,v dicts."""
    import eccodes

    index_keys = ['date', 'time', 'shortName', 'level', 'step']

    idx = eccodes.codes_index_new_from_file(infile.as_posix(), index_keys)
    values = None

    all_out = []
    # iterate date
    for date in eccodes.codes_index_get(idx, 'date'):
        eccodes.codes_index_select(idx, 'date', date)
        # iterate time
        for time in eccodes.codes_index_get(idx, 'time'):
            eccodes.codes_index_select(idx, 'time', time)
            for step in eccodes.codes_index_get(idx, 'step'):
                eccodes.codes_index_select(idx, 'step', step)

                out = get_values(idx, step, levelist=range(37, 138))
                hour = time[0] if len(time) == 3 else time[:2]
                ts = pd.Timestamp(year=int(date[:4]), month=int(date[4:6]),
                                  day=int(date[6:]),  hour=int(hour), tz='UTC')
                out['dt'] = ts
                all_out.append(out)

    return all_out


def _delete_grib_file(file_path):
    """Delete ECMWF file."""
    # delete the file:
    try:
        file_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (file_path, e.strerror))


def get_ecmwf_data(start_date, end_date, lat, lon, grib_dir=None, delete=False, need_z=False):
    """Download and process ECMWF forecast grib files to a pandas dataframe.

    Download ECMWF model level data, process the downloaded files, and save the
    forecast information to a pickled pandas dataframe.

    Note: will check for existing grib files for each month within date range,
    so will not redownload data that already exists on disc.

    Parameters
    ==========
    start_date : str
        Starting date of data to collect, in any format that pd.Timestamp will
        accept, for example 'YYYYMMDD'.
    end_date : str
        Ending date of data to collect, in any format that pd.Timestamp will
        accept, for example 'YYYYMMDD'.
    lat : float
        Latitude of site of interest, to nearest 0.25 degrees
    lon : float
        Longitude of site of interest, to nearest 0.25 degrees
    grib_dir : str
        Path to dir in which to save the raw grib files. Default is None, in
        which case grib files will be stored in the package data directory.
    delete : bool
        Whether to delete the grib files once the info has been reformatted.
        Default is False, so the files will be kept.

    """
    # download process is most efficient done monthly, so find all the dates
    # to iterate over to cover from desired start to end dates.
    dates = _get_iter_dates(start_date, end_date)
    f_template = 'ecmwf_{}_{}_{}_{}_uvt.grib'

    for m1, m2 in dates:
        grib_f = f_template.format(lat, lon,
                                   m1.strftime('%Y%m%d'),
                                   m2.strftime('%Y%m%d'))
        grib_path = pathlib.Path.joinpath(grib_dir, grib_f)
        # test if file exists
        if not grib_path.exists():
            # if not, download it!
            _download_ecmwf(m1.strftime('%Y-%m-%d'),
                            m2.strftime('%Y-%m-%d'),
                            lat, lon, grib_path, need_z)

    values_dict = []
    for m1, m2 in dates:
        # grib file name for each months data
        grib_f = f_template.format(lat, lon,
                                   m1.strftime('%Y%m%d'),
                                   m2.strftime('%Y%m%d'))
        grib_path = pathlib.Path.joinpath(grib_dir, grib_f)
        values_dict = values_dict + _process_grib(grib_path)

        if delete:
            _delete_grib_file(grib_path)

    df = pd.DataFrame(values_dict).set_index('dt')

    save_file = f'ecmwf_{lat}_{lon}_{start_date}_{end_date}.p'
    save_path = pathlib.Path.joinpath(grib_dir, save_file)
    pickle.dump(df, open(save_path, 'wb'))


class MissingLevelError(Exception):
    """Exception capturing missing levels in input."""
    pass


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-lat', type=float, default=-30.25)
    parser.add_argument('-lon', type=float, default=-70.75)
    parser.add_argument('-d1', type=str, default='20190501')
    parser.add_argument('-d2', type=str, default='20190531')
    parser.add_argument('-t', type=str, default='04/06/08')
    parser.add_argument('-grib_dir', type=str, default=None)
    parser.add_argument('--keep_grb', default=True, action='store_false')
    parser.add_argument('--get_z', default=False, action='store_true')
    args = parser.parse_args()

    # if no custom option defined, put the grib files in the data folder
    if args.grib_dir is None:
        grib_dir = PKG_DATA_DIR
    else:
        grib_dir = pathlib.Path(args.grib_dir)
    # move CWD to data directory for easier downloading/etc
    os.chdir(grib_dir)

    # call main function to execute the downloading/processing/saving
    get_ecmwf_data(args.t,
                   start_date=args.d1,
                   end_date=args.d2,
                   lat=args.lat,
                   lon=args.lon,
                   grib_dir=grib_dir,
                   delete=args.keep_grb,
                   need_z=args.get_z)

    if args.get_z:
        geopotential_on_ml(args.d1, args.lat, args.lon, grib_dir=grib_dir)
        if args.keep_grb:
            _delete_grib_file(pathlib.Path.joinpath(grib_dir, 'zlnsp_ml.grib'))
            _delete_grib_file(pathlib.Path.joinpath(grib_dir, 'tq_ml.grib'))
