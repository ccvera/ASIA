# ML METEO

Estructura del repositorio:
- **RaP**: contiene el fuente de la herramienta
- **data**: contiene datos en crudo de ejemplo para el uso de los scripts de preprocesamiento almacenados en "utils".
  - raw: datos en crudo obtenidos de las simulaciones del modelo WRF.
  - nc: scripts filtrados con las variables que necesitamos.
  - csv: ficheros ya procesados que conformarán el _dataset.
- **reports**: contienen informes con los resultados obtenidos por la herramienta.
- **utils**: conjunto de scripts para llevar a cabo el preprocesamiento de los datos.

## Prerequisitos
- Python 2.7
- wrf.python
- scikit-learn

## Preprocesamiento
```
$ cd utils
```

Para filtrar las salidas del WRF y obtener únicamente las variables que necesitamos:
```
 $ python filter_nc_variables.py -D <raw_data_dir> -O <output_nc_dir>
```

Para convertir nuetro .nc filtrado en un .csv:
```
$ python create_csv.py -D <output_nc_dir> -O <output_csv_di> -f datos_interpolados.nc
```

Donde el fichero datos interpolados, corresponde a la verdad terreno obtenida de las mediciones de los pluviómetros de la CHE.

Finalmente, obtenemos un .csv único, el cual almacenará toda la información:
```
$ python merge_csv.py -t <dataset_csv_train_dir> -v <dataset_csv_validation_dir>
```
