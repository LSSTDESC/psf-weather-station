import cdsapi

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


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d1', type=str, default='2019-05-01')
    parser.add_argument('-d2', type=str, default='2019-05-31')
    parser.add_argument('-outdir', type=str, default='../data/emcwf/')
    args = parser.parse_args()

    download(args)
