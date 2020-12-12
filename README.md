# psf-weather-station
Generate realistic wind and turbulence integral parameters for atmospheric point-spread function simulations.

This package uses two inputs: [NOAA Global Forecasting System (GFS)](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/global-forcast-system-gfs) outputs, and local wind measurements at the observatory. The repo contains these data for Cerro Pachon, and all defaults are set up to match this location, so generating parameters for Cerro Pachon (and Cerro Telolo, nearby) is straightforward. For use at other observatories, users must supply some input data: instructions for downloading NOAA datasets and formatting telemetry are below.

Note that the outputs match desired inputs for [GalSim](https://github.com/GalSim-developers/GalSim) atompsheric PSF simulations.

# installation and use
Code can currently be installed through github by running the following:

```
git clone https://github.com/cahebert/psf-weather-station.git
cd psf-weather-station
python setup.py install
```

In python, import as psfws:
`import psfws`

## requirements
numpy, pandas, pickle, scipy, pathlib

# customizing location
## formatting telemetry
The observatory telemetry should be in a dictionary, stored as a pickle file. The dictionary must have two keys: "wind_direction" and "wind speed". The values for each of these should be pandas (pd) Series holding measured values in degrees and m/s, respectively. The Series index for both should be pd.datetime objects of the measurement times, in UTC. 
## downloading new GFS data
The repository contains code to automatically download and process GFS data of interest for the user. The only inputs required are location of observatory, in lat/long rounded to nearest 0.5 degrees (e.g. for Gemini South, we use lat=-30 and lon=289.5), and start/end dates formatted as in 20190528 (ideally these should roughly match start/end dates of available telemetry).

Run:

`psfws.get_noaa_data(start_date, end_date, lat, lon)`

For example, for Cerro Pachon, one could run the following:

`psfws.get_noaa_data(20190501, 20190603, -30, 289.5)`

Or, from command line, one can run:

`python psf-weather-station/psfws/get_noaa_data.py -lat -30 -lon 289.5 -d1 20190501 -d2 20190603`

This will run the download and processing of NOAA GFS files: though it depends on how many months of data requested, this may take a long time (1-10h). The raw GFS data files will automatically be erased once the desired data has been extracted, so this should not take more than 60MB of disc space at any given time.

Additional requirements: pygrib

### aside: installing pygrib
Installing this package is not always easy, so for this reason it is not a dependency for most of psf-weather-station. However, it is a requirement for using get_noaa_data() for the processing of GFS data files. For installation of pygrib, I recommend using conda rather than pip:

`conda install -c conda-forge pygrib`

See [this gituhub issue](https://github.com/jswhit/pygrib/issues/115) for more installation debugging ideas if this doesn't work for you.
