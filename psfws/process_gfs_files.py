"""Process raw NOAA GFS files and extract u/v wind data."""
import pygrib
import numpy as np
import pandas as pd
from os import listdir
import pickle
import argparse


def load_uv(date='20190501', time='0000', latitude=-30, longitude=289.5):
    """Load U and V componenents of wind for GFS file given date and time."""
    # load in dataset (specific date/time given by args)
    try:
        grbs = pygrib.open(f'rawGFS/gfsanl_4_{date}_{time}_000.grb2')
    except FileNotFoundError:
        print(f'Could not find: rawGFS/gfsanl_4_{date}_{time}_000.grb2')
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
        approx_long  = [longitude-0.5, longitude+0.5]

        # extract u and v values at specificc lat/long for each altitude
        for i in range(1, 32):
            d, lat, lon = uwind[i].data(lat1=approx_lat[0], lat2=approx_lat[1], 
                                        lon1=approx_long[0], lon2=approx_long[1])
            u_values[i - 1] = d[np.where((lat == latitude) & (lon == longitude))][0]

            d, lat, lon = vwind[i].data(lat1=approx_lat[0], lat2=approx_lat[1], 
                                        lon1=approx_long[0], lon2=approx_long[1])
            v_values[i - 1] = d[np.where((lat == latitude) & (lon == longitude))][0]

        return u_values, v_values


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-lat', type=float)
    args.add_argument('-long', type=float)

    rawFiles = np.sort(listdir('./rawGFS/')) 
    timestamps = []
    value_dicts = []

    # each file corresponds to one datetime
    for f in rawFiles:
        if len(f) < 10:
            continue
        # get date and time from the filename string, make a timestamp
        d, t = f.split('.')[0].split('_')[2:4]
        try:
            timestamp = pd.Timestamp(f'{d}T{t[:2]}', tz='UTC')
        except ValueError:
            print(f'could not convert {d}T{t[:2]} to a valid Timestamp!')

        # load u/v info for this file
        out = load_uv(d, t, latitude=args.lat, longitude=args.long)

        if out is not None:
            value_dicts.append({'u': out[0], 'v': out[1]})
            timestamps.append(timestamp)

    # put all the u/v data into dataframe
    uv_df = pd.DataFrame(value_dicts, index=timestamps)

    # these are first and last dates in dataset, include in file save name
    d1 = rawFiles[0].split('.')[0].split('_')[2]
    d2 = rawFiles[-1].split('.')[0].split('_')[2]

    pickle.dump(uv_df, open(f'./uv_winds_lat{args.lat}_lon{args.long}_{d1}-{d2}.pkl', 'wb'))
