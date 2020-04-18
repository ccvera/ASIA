import os
import netCDF4
import pandas as pd
import numpy as np

import datetime as dt
import argparse

def create_csv(nc_filtrado,in_dir,out_dir):

	filtrado        = netCDF4.Dataset(in_dir + "/" + nc_filtrado, 'r')

	# Obtengo las variables del fichero filtrado (datos de prediccion)
        lat     = filtrado.variables['XLAT'][:]
        lon     = filtrado.variables['XLONG'][:]
	hei	= filtrado.variables['HGT'][:]
        rainc   = filtrado.variables['RAINC'][:]
	rainnc  = filtrado.variables['RAINNC'][:]
        time    = filtrado.variables['DATE'][:]

	#aux	= filtrado.variables['Qvapor0'][:]

        # Se define el numero de veces que vamos a repetir las coordenadas, que se va a repetir la fecha (fecha, x, y, z) y que se va a repetir todo
        repeat_coord    = time.shape[0]
        #repeat_date     = precip.shape[1]*precip.shape[2]
        #repeat_all      = aux.shape[1]
	repeat_date	= lat.shape[1]*lat.shape[2]
	repeat_all	= 1

        print(repeat_all)

        # Obtenemos los valores de las variables en un array 1D
        lat_x   	= lat.flatten()
        lon_x   	= lon.flatten()
	hei_x		= hei.flatten()
        rainc_x  	= rainc.flatten()
	rainnc_x        = rainnc.flatten()
        time_x  	= time[:].flatten()

        # Creamos las series de cada una de las variables
        lat_serie       = pd.Series(lat_x,name='XLAT')
        lon_serie       = pd.Series(lon_x,name='XLONG')
	hei_serie       = pd.Series(hei_x,name='HGT')
        rainc_serie     = pd.Series(rainc_x,name='RAINC')
	rainnc_serie    = pd.Series(rainnc_x,name='RAINNC')

	time_serie      = pd.Series(time_x,name='DATE').repeat(repeat_date*repeat_all).reset_index(drop=True)
	lat_serie	= pd.concat([lat_serie], axis=0).reset_index(drop=True)
	lon_serie	= pd.concat([lon_serie], axis=0).reset_index(drop=True)
	hei_serie	= pd.concat([hei_serie], axis=0).reset_index(drop=True)
	rainc_serie	= pd.concat([rainc_serie]*repeat_all, axis=0).reset_index(drop=True)
	rainnc_serie    = pd.concat([rainnc_serie]*repeat_all, axis=0).reset_index(drop=True)

	coordenadas	= pd.concat([lat_serie,lon_serie,hei_serie], axis=1)
	coordenadas	= pd.concat([coordenadas]*repeat_all, axis=0).reset_index(drop=True)

	final_serie     = pd.concat([time_serie,coordenadas,rainc_serie,rainnc_serie], axis=1)

	########################

        # Concatenamos todas las series respecto al eje de las X (en horizontal)
	print("Creando final_serie...")
	#final_serie     = pd.concat([final_serie,precip_serie], axis=1)

	# Eliminamos las filas para las que no tenemos los valores de la precip real
	print(final_serie.dropna(inplace=True))

        # Almacenamos en un csv
	csv_name	= nc_filtrado[0:10]
        final_serie.to_csv(out_dir + "/" + csv_name + '.csv',index=False,header=True, float_format='%.8f')

def create_csv_files(in_dir,out_dir):
	files = os.listdir(in_dir)
        files.sort()

        for i,nc_file in enumerate(files):
                print("Creando fichero .csv para el fichero...")
		print nc_file
		create_csv(nc_file,in_dir,out_dir)
def main():

	parser = argparse.ArgumentParser(description = "Description for my parser")
	parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF", required = False, default = "")
	parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_* [Default: ncfiles]", required = False, default = "ncfiles")
	parser.add_argument("-O", "--output_dir", help = "Directorio para los .csv de salida [Default: out_csv]", required = False, default = "outcsv")

	argument = parser.parse_args()

	if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
	if argument.output_dir:
            out_dir = format(argument.output_dir)
            if not os.path.exists(format(argument.output_dir)):
                os.makedirs(format(argument.output_dir))
        if argument.dir:
            print("You have used '-D' or '--dir' with argument: {0}".format(argument.dir))
            create_csv_files(format(argument.dir), out_dir)

if __name__ == "__main__":
	main()

