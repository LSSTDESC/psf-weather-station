import pygrib
import numpy as np
import pandas as pd
from os import listdir
import pickle


def load_uv(date='0501', time='0000', latitude=-30, longitude=289.5):
    '''
    Load the U and V componenents of wind for the GFS file
    corresponding to the given date and time.
    '''
    # load in this dataset (specific date/time)
    try:
        grbs = pygrib.open(f'rawGFS/gfsanl_4_2019{date}_{time}_000.grb2')
    except FileNotFoundError:
        print(f'Could not find: rawGFS/gfsanl_4_2019{date}_{time}_000.grb2')
        return None

    # select an example variable
    uwind = grbs.select(name='U component of wind')
    vwind = grbs.select(name='V component of wind')

    if len(uwind) < 32:
        print(f'file {date} {time} had incomplete wind information.')
        return None
    else:
        u_values = np.zeros(31)
        v_values = np.zeros(31)

        gust = grbs.select(name='Wind speed (gust)')[0]
        d, lat, lon = gust.data(lat1=-31, lat2=-29.5, lon1=289, lon2=290)
        gust_values = d[np.where((lat == latitude) & (lon == longitude))][0]

        for i in range(1, 32):
            d, lat, lon = uwind[i].data(lat1=-31, lat2=-29, lon1=289, lon2=290)
            u_values[i - 1] = d[np.where((lat == latitude) & (lon == longitude))][0]

            d, lat, lon = vwind[i].data(lat1=-31, lat2=-29, lon1=289, lon2=290)
            v_values[i - 1] = d[np.where((lat == latitude) & (lon == longitude))][0]
        return u_values, v_values, gust_values


rawFiles = np.sort(listdir('./rawGFS/'))[:20]
timestamps = []
value_dicts = []

for f in rawFiles:
    if len(f) < 10:
        continue
    # get date and time from the filename string, make a timestamp
    d, t = f.split('.')[0].split('_')[2:4]
    try:
        timestamp = pd.Timestamp(f'{d}T{t[:2]}', tz='UTC')
    except ValueError:
        print(f'could not convert {d}T{t[:2]} to a valid Timestamp!')

    latitude=-30.5
    longitude=289

    out = load_uv(d[4:], t, latitude=latitude, longitude=longitude)

    if out is not None:
        value_dicts.append({'u': out[0], 'v': out[1], 'gust': out[2]})
        timestamps.append(timestamp)

uv_df = pd.DataFrame(value_dicts, index=timestamps)
d1 = rawFiles[0].split('.')[0].split('_')[2][4:]
d2 = rawFiles[-1].split('.')[0].split('_')[2][4:]
pickle.dump(uv_df, open(f'./uv_winds_lat{latitude}_lon{longitude}_{d1}19-{d2}19.pkl', 'wb'))
