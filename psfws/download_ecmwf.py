"""Functions to download and process ECMWF forecast data."""

import numpy as np
import pandas as pd
import pickle
import pathlib
import os

PKG_BASE = pathlib.Path(__file__).resolve().parents[1].absolute()
DATA_DIR = pathlib.Path.joinpath(PKG_BASE, 'data/')


def _download_ecmwf(m1, m2, lat, lon, save_path):
    """Download ECMWF model level data.

    Note: best practice is to query month by month.
    """
    import cdsapi
    cds = cdsapi.Client()

    # check that the lat/lon values are %.25 = 0
    lat_lon_check = lat % 0.25 + lon % 0.25
    assert lat_lon_check == 0, "Lat, lon must be rounded to nearest 0.25 deg!"

    cds.retrieve('reanalysis-era5-complete', {
                  'type': 'an',  # 'an' for analysis
                  'grid': '0.25/0.25',  # need this to get lat/lon outputs!!
                  'area': f"{lat}/{lon}/{lat}/{lon}",  # N/W/S/E bounds
                  'time': '00/06/12/18',  # output times to download
                  'date': f"{m1}/to/{m2}",
                  'class': 'ea',
                  'param': '130/131/132',  # codes for temp, wind speeds
                  'expver': '1',
                  'format': 'grib',  # output file format
                  'stream': 'oper',
                  'levtype': 'ml',  # model level outputs (finely sampled in h)
                  'levelist': '37/to/137',  # don't need high altitude levels
                  }, save_path)

    return


def _get_month_edges(date):
    """Return Timestamps for start and end of the month of given date."""
    month_start = pd.Timestamp(year=date.year, month=date.month, day=1)
    next_month_start = pd.Timestamp(year=date.year, month=date.month+1, day=1)
    return month_start, next_month_start - pd.Timedelta(days=1)


def _get_iter_dates(start_date, end_date):
    """Return list of date pairs to iterate over months of interest."""
    d1 = pd.Timestamp(start_date)
    d2 = pd.Timestamp(end_date)

    # gonna assume for now that the years are the same
    if d1.month == d2.month:
        dates = [(d1, d2)]
    if d1.month == d2.month - 1:
        dates = [(d1, _get_month_edges(d1)[1]), (_get_month_edges(d2)[0], d2)]
    else:
        middle_months = np.arange(d1.month + 1, d2.month)
        # day doesn't matter but needed as argument
        middle_pairs = [_get_month_edges(pd.Timestamp(year=d1.year,
                                                      month=m, day=1))
                        for m in middle_months]

        dates = [(d1, _get_month_edges(d1)[1])] + middle_pairs + \
                [(_get_month_edges(d2)[0], d2)]

    return dates


def _process_grib(infile, t, u, v):
    """Process desired grib file, add data to t,u,v dicts."""
    with eccodes.GribFile(infile) as grib:
        for msg in grib:
            ts = pd.Timestamp(year=msg['year'], month=msg['month'],
                              day=msg['day'],  hour=msg['hour'], tz='UTC')
            for var, var_dict in zip(['T', 'U', 'V'], [t, u, v]):
                if var in msg['name']:
                    if ts in var_dict.keys():
                        var_dict[ts].append(msg['values'])
                    else:
                        var_dict[ts] = [msg['values']]


def _delete_grib_file(file_name):
    """Delete ECMWF file."""
    file_path = pathlib.Path.joinpath(DATA_DIR, file_name)

    # delete the file:
    try:
        file_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (file_path, e.strerror))


def get_ecmwf_data(start_date, end_date, lat, lon, grib_dir, delete):
    """Download and process ECMWF forecast grib files to a pandas dataframe."""
    import eccodes

    # download process is most efficient done monthly, so find all the dates
    # to iterate over to cover from desired start to end dates.
    dates = _get_iter_dates(start_date, end_date)
    f_template = 'ecmwf_{}_{}_{}_{}_uvt.grib'

    # if no custom option defined, put the grib files in the data folder
    if grib_dir is None:
        grib_dir = DATA_DIR

    for m1, m2 in dates:
        grib_f = f_template.format(lat, lon,
                                   m1.strftime('%Y-%m-%d'),
                                   m2.strftime('%Y-%m-%d'))
        grib_path = pathlib.Path.joinpath(grib_dir, grib_f)
        # test if file exists
        if not grib_path.exists():
            # if not, download it!
            _download_ecmwf(m1.strftime('%Y-%m-%d'), m2.strftime('%Y-%m-%d'),
                            lat, lon, grib_path)

    t, u, v = {}, {}, {}

    for m1, m2 in dates:
        # grib file name for each months data
        grib_f = f_template.format(lat, lon,
                                   m1.strftime('%Y-%m-%d'),
                                   m2.strftime('%Y-%m-%d'))
        _process_grib(grib_f, t, u, v)

        if delete:
            _delete_grib_file(grib_f)

    timestamps = t.keys()
    values_dict = [{'t': t[ts], 'u': u[ts], 'v':v[ts]} for ts in timestamps]
    tuv_df = pd.DataFrame(values_dict, index=timestamps)

    save_file = f'ecmwf_{lat}_{lon}_{start_date}_{end_date}.grib'
    save_path = pathlib.Path.joinpath(grib_dir, save_file)
    pickle.dump(tuv_df, open(save_path, 'wb'))


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-lat', type=float, default=-30.25)
    parser.add_argument('-long', type=float, default=-70.75)
    parser.add_argument('-d1', type=str, default='20190501')
    parser.add_argument('-d2', type=str, default='20190531')
    parser.add_argument('-grib_dir', type=str, default=None)
    parser.add_argument('--keep_grb', type=bool, default=True,
                        action='store_false')
    args = parser.parse_args()

    # move CWD to data directory for easier downloading/etc
    os.chdir(DATA_DIR)

    get_ecmwf_data(args.d1, args.d2, args.lat, args.lon, args.grib_dir,
                   args.keep_grb)
