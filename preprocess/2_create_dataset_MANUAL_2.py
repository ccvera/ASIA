import os
import netCDF4
import pandas as pd
import numpy as np

import datetime as dt
import argparse

def create_csv(nc_filtrado,nc_interpolado):

	#filtrado	= netCDF4.Dataset(nc_filtrado, 'r', format='NETCDF4')
	filtrado        = netCDF4.Dataset(nc_filtrado, 'r')
	interpolado	= netCDF4.Dataset(nc_interpolado, 'r', format='NETCDF4')

	# Obtengo las variables del fichero filtrado (datos de prediccion)
        lat     = filtrado.variables['lat'][:]
        lon     = filtrado.variables['lon'][:]
	hei	= filtrado.variables['height'][:]
        rain    = filtrado.variables['precip'][:]
        time    = filtrado.variables['date'][:]

	aux	= filtrado.variables['Qvapor0'][:]

	# Obtengo las variables del fichero interpolado (nuestros datos reales)
	# Se define como fecha de inicio 2489 porque corresponde con el primer dia de las predicciones
	inicio	= 2589
	fin	= inicio + time.shape[0]
	precip	= interpolado.variables['precip'][inicio:fin,:,:]	
	print(precip.shape)
	print(rain.shape)

        # Se define el numero de veces que vamos a repetir las coordenadas, que se va a repetir la fecha (fecha, x, y, z) y que se va a repetir todo
        repeat_coord    = time.shape[0]
        repeat_date     = precip.shape[1]*precip.shape[2]
        repeat_all      = aux.shape[1]

        print(repeat_all)

        # Obtenemos los valores de las variables en un array 1D
        lat_x   	= lat.flatten()
        lon_x   	= lon.flatten()
	hei_x		= hei.flatten()
        rain_x  	= rain.flatten()
        time_x  	= time[:].flatten()

	precip_x	= precip[:].flatten()

        # Creamos las series de cada una de las variables
        lat_serie       = pd.Series(lat_x,name='XLAT')
        lon_serie       = pd.Series(lon_x,name='XLONG')
	hei_serie       = pd.Series(hei_x,name='HEIGHT')
        rain_serie      = pd.Series(rain_x,name='XRAIN')
        precip_serie    = pd.Series(precip_x,name='precip_real')

	time_serie      = pd.Series(time_x,name='DATE').repeat(repeat_date*repeat_all).reset_index(drop=True)
	lat_serie	= pd.concat([lat_serie]*repeat_coord, axis=0).reset_index(drop=True)
	lon_serie	= pd.concat([lon_serie]*repeat_coord, axis=0).reset_index(drop=True)
	hei_serie	= pd.concat([hei_serie]*repeat_coord, axis=0).reset_index(drop=True)
	rain_serie	= pd.concat([rain_serie]*repeat_all, axis=0).reset_index(drop=True)

	precip_serie	= pd.concat([precip_serie]*repeat_all, axis=0).reset_index(drop=True)

	coordenadas	= pd.concat([lat_serie,lon_serie,hei_serie], axis=1)
	coordenadas	= pd.concat([coordenadas]*repeat_all, axis=0).reset_index(drop=True)

	final_serie     = pd.concat([time_serie,coordenadas,rain_serie], axis=1)

	for i in range(0,24):
		name_qvapor	= "qv" + str(i)
		name_qrain	= "qr" + str(i)
		name_qcloud	= "qc" + str(i)

		var_qv,var_qr,var_qc 	= get_var(filtrado,name_qvapor,name_qrain,name_qcloud,i)

		name_qvapor_x	= name_qvapor + "_x"
                name_qrain_x   	= name_qrain + "_x"
                name_qcloud_x   = name_qcloud + "_x"

		name_qvapor_x	= var_qv.flatten()
		name_qrain_x	= var_qr.flatten()
                name_qcloud_x   = var_qc.flatten()

                serie_qvapor    = str(name_qvapor) + "_serie"
                serie_qrain     = str(name_qrain) + "_serie"
                serie_qcloud    = str(name_qcloud) + "_serie"

		serie_qvapor	= pd.Series(name_qvapor_x,name=name_qvapor)
                serie_qrain	= pd.Series(name_qrain_x,name=name_qrain)
                serie_qvapor    = pd.Series(name_qcloud_x,name=name_qcloud)
	
		final_serie     = pd.concat([final_serie,serie_qvapor,serie_qrain,serie_qvapor], axis=1)
		#final_serie     = pd.concat([final_serie,serie_qvapor,serie_qrain], axis=1)

	########################

        # Concatenamos todas las series respecto al eje de las X (en horizontal)
	print("Creando final_serie...")
	final_serie     = pd.concat([final_serie,precip_serie], axis=1)

	# Eliminamos las filas para las que no tenemos los valores de la precip real
	print(final_serie.dropna(inplace=True))

        # Almacenamos en un csv
        final_serie.to_csv('dataset_final.csv',index=False,header=True, float_format='%.8f')

def get_var(filtrado,var_qv,var_qr,var_qc,idx):
	qvapor		= "Qvapor" + str(idx)
	qrain		= "Qrain" + str(idx)
	qcloud		= "Qcloud" + str(idx)
	var_qv		= filtrado.variables[qvapor][:]
        var_qr          = filtrado.variables[qrain][:]
        var_qc          = filtrado.variables[qcloud][:]

	return var_qv,var_qr,var_qc

def main():

	#parser = argparse.ArgumentParser(description = "Description for my parser")
	#parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF", required = False, default = "")
	#parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_* [Default: ncfiles]", required = False, default = "ncfiles")
	#parser.add_argument("-f", "--file", help = "Archino wrfout_*", required = False, default = "")

	#argument = parser.parse_args()

	#if argument.Help:
        #    print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
	#if argument.dir:
	    #print("You have used '-D' or '--dir' with argument: {0}".format(argument.dir))

	filtrado	= "/home/fcsc/ccalvo/METEO/preproces/wrf_outputs_filtered.nc"
	interpolado	= "/home/fcsc/ccalvo/METEO/preproces/datos_interpolados.nc"
	create_csv(filtrado,interpolado)

if __name__ == "__main__":
	main()

