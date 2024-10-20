"""Functions to download and reformat ECMWF forecast data."""

import numpy as np
import pandas as pd
import pickle
import pathlib
import os
import utils

DATA_DIR = utils.get_data_path()


def _download_ecmwf(m1, m2, lat, lon, save_path):
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
                  'type': 'an',  # 'an' for analysis
                  'grid': '0.25/0.25',  # need this to get lat/lon outputs!!
                  'area': f"{lat}/{lon}/{lat}/{lon}",  # N/W/S/E bounds
                  'time': '00/06/18',  # output times to download
                  'date': f"{m1}/to/{m2}",
                  'class': 'ea',
                  'param': '130/131/132',  # codes for temp, wind speed comps
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
        dates = _get_iter_months(d1, y1_end)

        for y in np.arange(d1.year + 1, d2.year):
            y_start = pd.Timestamp(year=y, month=1, day=1)
            y_end = pd.Timestamp(year=y, month=12, day=31)
            dates += _get_iter_months(y_start, y_end)

        y2_start = pd.Timestamp(year=d2.year, month=1, day=1)
        dates += _get_iter_months(y2_start, d2)

    return dates


def _process_grib(infile):
    """Open downloaded grib file, add data to t,u,v DataFrame with Timestamp index."""
    import xarray as xr
    ds = xr.load_dataset(infile)
    data = pd.DataFrame([{'t': t.values.flatten(), 
                          'u': u.values.flatten(),
                          'v': v.values.flatten()} for t,u,v in zip(ds.t, ds.u, ds.v)], 
                        index=[pd.Timestamp(t.values, tz='UTC') for t in ds.time])
    ds.close()
    return data


def _delete_grib_file(file_path):
    """Delete ECMWF file."""
    # delete the file:
    try:
        file_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (file_path, e.strerror))


def get_ecmwf_data(start_date, end_date, lat, lon, grib_dir=None, delete=False):
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
                                   m1.strftime('%Y-%m-%d'),
                                   m2.strftime('%Y-%m-%d'))
        grib_path = pathlib.Path.joinpath(grib_dir, grib_f)
        # test if file exists
        if not grib_path.exists():
            # if not, download it!
            _download_ecmwf(m1.strftime('%Y-%m-%d'), m2.strftime('%Y-%m-%d'),
                            lat, lon, grib_path)

    df_list = []

    for m1, m2 in dates:
        # grib file name for each months data
        grib_f = f_template.format(lat, lon,
                                   m1.strftime('%Y-%m-%d'),
                                   m2.strftime('%Y-%m-%d'))
        grib_path = pathlib.Path.joinpath(grib_dir, grib_f)
        df_list.append(_process_grib(grib_path.as_posix()))

        if delete:
            _delete_grib_file(grib_path)

    tuv_df = pd.concat(df_list)

    save_file = f'ecmwf_{lat}_{lon}_{start_date}_{end_date}.p'
    save_path = pathlib.Path.joinpath(grib_dir, save_file)
    
    with open(save_path, 'wb') as f:
        pickle.dump(tuv_df, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-lat', type=float, default=-30.25)
    parser.add_argument('-lon', type=float, default=-70.75)
    parser.add_argument('-d1', type=str, default='20190501')
    parser.add_argument('-d2', type=str, default='20190531')
    parser.add_argument('-grib_dir', type=str, default=None)
    parser.add_argument('--keep_grb', default=True, action='store_false')
    args = parser.parse_args()
    
    # if no custom option defined, put the grib files in the data folder
    if args.grib_dir is None:
        grib_dir = pathlib.Path(DATA_DIR)
    else:
        grib_dir = pathlib.Path(args.grib_dir)
    # move CWD to data directory for easier downloading/etc
    os.chdir(grib_dir)

    # call main function to execute the downloading/processing/saving
    get_ecmwf_data(start_date=args.d1,
                   end_date=args.d2,
                   lat=args.lat,
                   lon=args.lon,
                   grib_dir=grib_dir,
                   delete=args.keep_grb)
