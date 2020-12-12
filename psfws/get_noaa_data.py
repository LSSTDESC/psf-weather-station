"""Functions to download and process NOAA GFS data."""

import os
import numpy as np
import pandas as pd
import pickle
import pathlib
from datetime import timedelta, datetime


url = 'https://www.ncei.noaa.gov/thredds/fileServer/' + \
      'model-gfs-g4-anl-files-old/'
URL_BASE = pathlib.Path(url)

PKG_BASE = pathlib.Path(__file__).resolve().parents[1].absolute()
DATA_DIR = pathlib.Path.joinpath(PKG_BASE, 'data/')


def _download_gfs_file(date, time):
    """Download a single GFS file according to date/time provided."""
    remote_path = pathlib.Path.joinpath(URL_BASE, f'{date[:6]}/{date}/')
    gfs_file = f'gfsanl_4_{date}_{time}_000.grb2'

    dl_link = pathlib.Path.joinpath(remote_path, gfs_file)
    dl_local = pathlib.Path.joinpath(DATA_DIR, gfs_file)

    # download the thing!
    print(dl_link.as_posix())
    os.system(f"curl -O {dl_link.as_posix()}")


def _delete_gfs_file(date, time):
    """Delete single GFS file according to date/time provided."""
    file_name = f'gfsanl_4_{date}_{time}_000.grb2'
    file_path = pathlib.Path.joinpath(DATA_DIR, file_name)

    # delete the file:
    try:
        file_path.unlink()
    except OSError as e:
        print("Error: %s : %s" % (file_path, e.strerror))


def _load_uv(date, time, latitude=-30, longitude=289.5):
    """Load U and V componenents of wind for GFS file given date and time."""
    # do not want pygrib to be a dependency for the whole psfws package.
    import pygrib

    # load in dataset (specific date/time given by args)
    try:
        grbs = pygrib.open(f'gfsanl_4_{date}_{time}_000.grb2')
    except FileNotFoundError:
        print(f'Could not find: gfsanl_4_{date}_{time}_000.grb2')
        return None

    # select relevant variables
    uwind = grbs.select(name='U component of wind')
    vwind = grbs.select(name='V component of wind')

    # check whether wind information exists in file
    if len(uwind) < 32:
        print(f'file {date} {time} had incomplete wind information.')
        return None
    else:
        u_values = np.zeros(31)
        v_values = np.zeros(31)

        # set location range
        approx_lat = [latitude-0.5, latitude+0.5]
        approx_long = [longitude-0.5, longitude+0.5]

        # extract u and v values at specificc lat/long for each altitude
        for i in range(1, 32):
            d, lat, lon = uwind[i].data(lat1=approx_lat[0],
                                        lat2=approx_lat[1],
                                        lon1=approx_long[0],
                                        lon2=approx_long[1])
            u_values[i-1] = d[np.where((lat == latitude) &
                                       (lon == longitude))][0]

            d, lat, lon = vwind[i].data(lat1=approx_lat[0],
                                        lat2=approx_lat[1],
                                        lon1=approx_long[0],
                                        lon2=approx_long[1])
            v_values[i-1] = d[np.where((lat == latitude) &
                                       (lon == longitude))][0]

        return u_values, v_values


def _datetime_range(start_date, end_date):
    """Datetime generator, outputs in 6hr increments bewteen input dates."""
    start_date = datetime(int(start_date[:4]), int(start_date[4:6]),
                          int(start_date[6:]))
    end_date = datetime(int(end_date[:4]), int(end_date[4:6]),
                        int(end_date[6:]))

    for n in range(int((end_date - start_date).days)):
        for hour in [0, 6, 12, 18]:
            yield start_date + timedelta(days=n, hours=hour)


def _get_date_and_time_strings(date):
    """Return string of YYYYMMDD and time in HHMM format for file access."""
    y, m, d, h = [str(i) for i in [date.year, date.month, date.day, date.hour]]

    # add zeros if needed
    if len(m) == 1:
        m = '0' + m
    if len(d) == 1:
        d = '0' + d
    if len(h) == 1:
        h = '0' + h
    if len(h) == 2:
        h += '00'

    return y+m+d, h


def get_noaa_data(start_date, end_date, lat, lon):
    """Download and process NOAA GFS data to pickle file for psfws."""
    # make start_date and end_date into datetime objects

    # where to store results:
    timestamps = []
    value_dicts = []

    # iterate through these in for loop
    for date in _datetime_range(start_date, end_date):
        try:
            timestamp = pd.Timestamp(date, tz='UTC')
        except ValueError:
            print(f'could not convert file to a valid Timestamp!')
            continue

        # get strings of date/time to interface with paths more easily
        d, h = _get_date_and_time_strings(date)

        # download file:
        _download_gfs_file(d, h)

        # process to extract u/v wind info
        out = _load_uv(d, h, lat, lon)

        if out is not None:
            value_dicts.append({'u': out[0], 'v': out[1]})
            timestamps.append(timestamp)

        # no longer need this file, delete to save disc space:
        _delete_gfs_file(d, h)

    # put all the u/v data into dataframe
    uv_df = pd.DataFrame(value_dicts, index=timestamps)

    save_file = f'gfs_{lat}_{lon}_{start_date}-{end_date}.pkl'
    save_path = pathlib.Path.joinpath(DATA_DIR, save_file)

    pickle.dump(uv_df, open(save_file, 'wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-lat', type=float)
    parser.add_argument('-long', type=float)
    parser.add_argument('-d1', type=str)
    parser.add_argument('-d2', type=str)
    args = parser.parse_args()

    # move CWD to data directory for easier downloading/etc
    os.chdir(DATA_DIR)

    get_noaa_data(args.d1, args.d2, args.lat, args.long)
