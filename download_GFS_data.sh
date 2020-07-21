#!/bin/bash

month='10'
for day in '01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' '23' '24' '25' '26' '27' '28' '29' '30' '31'
do
for h in '0000' '0600' '1200' '1800'
do
dl_link='https://www.ncei.noaa.gov/thredds/fileServer/model-gfs-g4-anl-files/2019'$month'/2019'$month$day'/gfsanl_4_2019'$month$day'_'$h'_000.grb2'
curl -O $dl_link

done
done
