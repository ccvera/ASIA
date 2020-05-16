# ML METEO

Para filtrar las salidas del WRF y obtener únicamente las variables que necesitamos:
```
 python 1_preprocess_daily.py -D <raw_data_dir> -O <output_nc_dir>
```

Para convertir nuetro .nc filtrado en un .csv:
```
python 2_create_dataset_daily.py -D <output_nc_dir> -O <output_csv_di> -f datos_interpolados.nc
```

Donde el fichero datos interpolados, corresponde a la verdad terreno.
