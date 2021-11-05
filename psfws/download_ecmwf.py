"""Functions to download and process ECMWF forecast data."""
import cdsapi
import numpy as np
import eccodes
import pandas as pd
import pickle

def download(args):
    """Download ECMWF model level data. Note: best practice is to query month by month."""
    cds = cdsapi.Client()

    outfile = args.outdir + f'{args.d1}-{args.d2}_uvt_cp.grib'

    cds.retrieve('reanalysis-era5-complete', {
                  'type': 'an', #'an' for analysis
                  'grid': '0.25/0.25', # need this to get lat/lon outputs!!
                  'area': '-30.25/-70.75/-30.25/-70.75', # N/W/S/E bounds
                  'time': '00/06/12/18', # output times to download
                  'date': f"{args.d1}/to/{args.d2}",
                  'class': 'ea', 
                  'param': '130/131/132', # these code for temp and wind speeds (in some order)
                  'expver': '1',
                  'format': 'grib', # output file format
                  'stream': 'oper',
                  'levtype': 'ml', # model level outputs (finely sampled in h)
                  'levelist': '37/to/137', # don't need the high altitude levels.
                  }, outfile)

    return

def get_month_edges(date):
    """Return Timestamps for start and end of the month of given date."""
    month_start = pd.Timestamp(year=date.year, month=date.month, day=1)
    next_month_start = pd.Timestamp(year=date.year, month=date.month+1, day=1)
    return month_start, next_month_start - pd.Timedelta(days=1)

def get_iter_dates(d1, d2):
    """Return list of date pairs to iterate over months of interest."""
    ## gonna assume for now that the years are the same
    if d1.month == d2.month:
        dates = [(d1, d2)]
    if d1.month == d2.month - 1:
        dates = [(d1, get_month_edges(d1)[1]), (get_month_edges(d2)[0], d2)]
    else:
        middle_months = np.arange(d1.month + 1, d2.month)
        # day doesn't matter but needed as argument
        middle_pairs = [get_month_edges(pd.Timestamp(year=d1.year, 
                                                     month=m, day=1)) 
                        for m in middle_months]

        dates = [(d1, get_month_edges(d1)[1])] + middle_pairs + \
                [(get_month_edges(d2)[0], d2)]

    return dates

def main(args):
    """Process grib files for given dates and save resulting dataframe."""
    # get list of Timestamps corresponding to 
    dates = get_iter_dates(pd.Timestamp(args.d1), pd.Timestamp(args.d2))
    
    t = {}
    u = {}
    v = {}
    
    for m1, m2 in dates:
        with eccodes.GribFile(args.f.format(m1.strftime('%Y-%m-%d'), 
                                            m2.strftime('%Y-%m-%d'))) as grib:
            for msg in grib:
                ts = pd.Timestamp(year=msg['year'], month=msg['month'], 
                                  day=msg['day'],  hour=msg['hour'], tz='UTC')
                for var, var_dict in zip(['T','U','V'], [t,u,v]):
                    if var in msg['name']:
                        if ts in var_dict.keys():
                            var_dict[ts].append(msg['values'])
                        else:
                            var_dict[ts] = [msg['values']]

    timestamps = t.keys()
    values_dict = [{'t':t[ts], 'u': u[ts], 'v':v[ts]} for ts in timestamps]
    tuv_df = pd.DataFrame(values_dict, index=timestamps)   

    pickle.dump(tuv_df, open(args.savef.format(args.d1,args.d2), 'wb'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d1', type=str, default='2019-05-01')
    parser.add_argument('-d2', type=str, default='2019-05-31')
    parser.add_argument('-outdir', type=str, default='../data/emcwf/')
    parser.add_argument('-f', type=str, 
                        default='../data/emcwf/{}-{}_uvt_cp.grib')
    parser.add_argument('-savef', type=str,
                        default='../data/emcwf/{}_{}_uvt.p')

args = parser.parse_args()

    download(args)
