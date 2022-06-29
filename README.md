[![test](https://github.com/LSSTDESC/psf-weather-station/actions/workflows/test.yaml/badge.svg)](https://github.com/LSSTDESC/psf-weather-station/actions/workflows/test.yaml)

# psf-weather-station
Leverage weather forecasting data to produce realistically correlated wind and turbulence parameters for atmospheric point-spread function  (PSF) simulations.

The weather forecasting data used in this package are the outputs of global circulation models (GCM). `psf-weather-station` can produce environment parameters for specific date/times, if configured with the appropriate input data. Included in the install are two example datasets for the months of May through October 2019, at the latitude/longitude point closest to Cerro Pachon. These are reanalysis data (i.e. forecasts retroactively analyzed with assimmilated measurements) from two different GCM: [NOAA Global Forecasting System (GFS) Analysis](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs) and [European Center for Medium-range Weather Forecasting (ECMWF) ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) (recommended). See instructions below on downloading data from these models for different dates and/or geographical locations.

The weather forecasting data provide the time-dependence desired for simulation parameters; some information about the distribution of turbulence is required to set the statistics of seeing. Specifically, these are the `s` and `scale` parameters of a log-normal PDF for each of the ground and free atmosphere turbulence integrals $J$. 
For example, the quantiles of cumulative MASS/DIMM measurements from Cerro Pachon in Table 1 of [Tokovinin and Travouillon 2006](https://ui.adsabs.harvard.edu/link_gateway/2006MNRAS.365.1235T/doi:10.1111/j.1365-2966.2005.09813.x) can be fit to such a distribution. An approximate conversion of the seeing quantiles at Cerro Telolo (Table 1 of [Tokovinin et al 2003](https://academic.oup.com/mnras/article/340/1/52/1130015)) or at Mauna Kea (Table 3 of [Tokovinin et al 2005](https://www.jstor.org/stable/10.1086/428930)) could likewise be used.

Note that the parameterization of outputs match the desired inputs for [GalSim](https://github.com/GalSim-developers/GalSim) atompsheric PSF simulations, including transforming parameters to relevant sky coordinates given the alt/az of an observation (see notebooks/demo.ipynb for more information).

### Optional additional data
- Observatory telemetry can provide more robust estimates of ground wind speeds and directions. To include such data, they should be saved as a pickled dictionary with three keys: `"wind_direction", "wind_speed", "temperature"`. The values for each of these should be a pandas `pd.Series` holding measured values in units of degrees, m/s, and Kelvin, respectively. The index for all three should be `pd.datetime` objects of the measurement times, in UTC. Note: for these to be included successfully, the telemetry should span the same date range as the forecast data.
- There is inconclusive evidence that turbulence strength at the ground may be correlated to both wind direction and speed; it is optional to specify a correlation coefficient between either, or both, of these pairs to include such an effect in the output parameters.

# installation and use
Code can currently be installed through github by running the following:

```
git clone https://github.com/cahebert/psf-weather-station.git
cd psf-weather-station
python setup.py install
```

In Python, import as psfws:
`import psfws`

## requirements
`numpy, pandas, pickle, scipy, pathlib`

## usage
See the `notebooks/demo.ipynb` notebook for an example of running an atmospheric PSF simulation using parameters from `psf-weather-station`.

## downloading weather forecasting data
The repository contains code to easily download and process GFS and/or ERA5 data of interest for the user.

Additional packages required for this functionality:
- for ECMWF ERA5, `cdsapi` and `eccodes`
- for NOAA GFS, `pygrib` (see aside below)

Note on choice of ERA5 vs GFS; for dates prior to Feb 2021, GFS has a much coarser sampling of the atmosphere, so it is recomended that ERA5 be used. The default GFS data acquisition in this package is for the dates prior to 2021.

Inputs required for the download are:
- `lat, lon`: location of observatory, in latitude/longitude rounded to nearest 0.25 degrees (e.g. for Gemini South, we use `lat=-30.25` and `lon=289.25`)
- `start_date, end_date`: dates bookending the desired data, formatted as "YYYYMMDD". 

Usage is shown below; simply substitute "noaa" for "ecmwf" in the function name to download GFS data instead. 

From inside a python script, run:

`psfws.get_ecmwf_data(start_date, end_date, lat, lon)`

Or, from command line, one can run:

`python psf-weather-station/psfws/get_ecmwf_data.py -lat lat -lon lon -d1 start_date -d2 end_date`

For example, for Cerro Pachon, one could run either of the following:

`psfws.get_ecmwf_data("20190501", "20190603", -30.25, 289.25)`

`python psf-weather-station/psfws/get_ecmwf_data.py -lat -30.25 -lon 289.25 -d1 "20190501" -d2 "20190603"`


This will run the download and processing of files. The final processed data are saved in a pickle file in the `psf-weather-station/psfws/data/` directory.
- ERA5: The Copernicus Data Server (cds) server which hosts the ECMWF data only returns specific columns of data requested, and works most efficiently with one request per month of data, so the downloaded files are not large. Because of the column selection, though, requests sometimes queue for ~hours. The raw data files, after being processed and relevant information saved, are deleted as default. 
- GFS: Though it depends on how many months of data requested, this download may take a long time (1-10h). The raw GFS data files, one per date/time containing all available columns, will automatically be erased once the desired data has been extracted so this should not take more than 60MB of disc space at any given time. 

### aside: installing pygrib
Installing this package is not always easy, so for this reason it is not a dependency for most of `psf-weather-station`. However, it is a requirement for using `get_noaa_data()` for the processing of GFS data files. It is *not* required for ECMWF ERA5 downloads. For installation of `pygrib`, I recommend using conda rather than pip:

`conda install -c conda-forge pygrib`

See [this gituhub issue](https://github.com/jswhit/pygrib/issues/115) for more installation debugging ideas if this doesn't work for you.
