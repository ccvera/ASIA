import os
import netCDF4
import pandas as pd
import numpy as np

import datetime as dt
import argparse
from optparse import OptionParser, Values

def create_dataset(datos_reales, datos_predic):

	datos_reales	= netCDF4.Dataset(datos_reales, 'r', format='NETCDF4')
	datos_predic	= netCDF4.Dataset(datos_predic, 'r', format='NETCDF4')

        # Obtenemos las variables del fichero netcdf
        lat     	= datos_predic.variables['lat'][:]
        lon     	= datos_predic.variables['lon'][:]
	precip_real	= datos_reales.variables['precip'][:]
	precip_predic	= datos_predic.variables['precip'][:]

        # Creamos el fichero .nc de datos
        dataset = netCDF4.Dataset('dataset.nc','w', format='NETCDF4')

        # Dimensiones
        dim_time, dim_lat, dim_lon  = precip_predic.shape
        print("Las dimensiones de la precipitacion son (time,lat,lon):")
        print(dim_time)
        print(dim_lat)
        print(dim_lon)
        print("--------------")

	print(precip_real.shape)
	precip_real   = datos_predic.variables['precip'][0:dim_time,:,:]

        # Configuramos nuestro nuevo ficher nc 
        # Definimos las dimensiones que tendran las distintas variables
        dataset.createDimension('lat',dim_lat)
        dataset.createDimension('lon',dim_lon)
        dataset.createDimension('time',dim_time)

        latitude        = dataset.createVariable('lat','f4',('lat','lon'))
        longitude       = dataset.createVariable('lon','f4',('lat','lon'))
        rainc_real	= dataset.createVariable('precip_real','f4',('time','lat','lon'))
	rainc_predic	= dataset.createVariable('precip_predicha','f4',('time','lat','lon'))

        # Datos a las variables
        latitude[:]     = lat
        longitude[:]    = lon
	rainc_real[:]	= precip_real
	rainc_predic[:]	= precip_predic

        dataset.close()


def get_accumulated_rain(parent_dir,directory):
	files = os.listdir(parent_dir + directory)
	files.sort()

	for i,f in enumerate(files):
		print(f)
		nc	= netCDF4.Dataset(parent_dir + directory + "/" + f, 'r', format='NETCDF4')
		rainc   = nc.variables['RAINC'][0,:,:]
		rainnc  = nc.variables['RAINNC'][0,:,:]

		if i == 0:
			rain = rainc + rainnc
		else:
			rain    += rainc + rainnc

	return rain

def get_date(parent_dir,directory):

	files = os.listdir(parent_dir + directory)
	files.sort()

	nc      = netCDF4.Dataset(parent_dir + directory + "/" + files[0], 'r', format='NETCDF4')
	time	= nc.variables['XTIME']
	
	#print(time[0])

	dtime   	= netCDF4.num2date(time[:],time.units)
	#print(dtime)
	str_time        = [i.strftime("%Y-%m-%d") for i in dtime]

	print("Obteniendo fecha...")
	print(str_time)
	
	# Convertimos la lista a un string para que lo acepte netCDF
	str1 = " "     
	return (str1.join(str_time)) 


def get_global_variables(parent_dir,folders):
	# Obtenemos las variables importantes
	print("Obteniendo valores de latitud y longitud del fichero...")
	f = "/home/fcsc/ccalvo/METEO/preproces/nc_data/2015020118/wrfout_d02_2015-02-02_01:00:00"

        # Leemos el primer fichero nc dentro del directorio "ncfiles" (asumimos que todos son del mismo dominio)
        nc      = netCDF4.Dataset(f, 'r', format='NETCDF4')

        # Obtenemos las variables del fichero netcdf
        lat     = nc.variables['XLAT'][0,:,:]		# Latitud
        lon     = nc.variables['XLONG'][0,:,:]		# Longitud
	hei	= nc.variables['HGT'][0,:,:]		# Altitud
        time    = nc.variables['XTIME']			# Hora

	Q_v	= nc.variables['QVAPOR']		# Vapor de agua

	# PRUEBAS CON NIVELES DE PRESION-----------------
	t	= nc.variables['T']			# Temperatura
	press	= nc.variables['PB']
	print(press.shape)

	lev_850	= np.where(nc.variables['PB'][:] == 700*100)
	print (lev_850)
	#t_850	= t[lev_850]

	#print(t_850)
	#-----------------------------------------------

        # Creamos las series de cada una de las variables
        dtime           = netCDF4.num2date(time[:],time.units)
        str_time        = [i.strftime("%Y-%m-%d") for i in dtime]

        # Creamos el fichero .nc de datos
	#Nota: quito el formato para poder utilizar strings en la fecha
        #nc_new  = netCDF4.Dataset(str_time[0] + '_filtered.nc','w', format='NETCDF4')
	nc_new  = netCDF4.Dataset('wrf_outputs_filtered.nc','w')

        # Dimensiones
        dim_lat, dim_lon  = lat.shape
	dim_time = len(folders)
	dim_hour = 1
	dim_alt =41
        print("Las dimensiones de la precipitacion son (time,lat,lon):")
        print(dim_time)
        print(dim_lat)
        print(dim_lon)
	print(dim_alt)
        print("--------------")

        # Configuramos nuestro nuevo ficher nc 
        # Definimos las dimensiones que tendran las distintas variables
        nc_new.createDimension('lat',dim_lat)
        nc_new.createDimension('lon',dim_lon)
        nc_new.createDimension('time',dim_time)
	nc_new.createDimension('hour',dim_hour)
        nc_new.createDimension('alt',dim_alt)

        latitude        = nc_new.createVariable('lat','f4',('lat','lon'))
        longitude       = nc_new.createVariable('lon','f4',('lat','lon'))
	height		= nc_new.createVariable('height','f4',('lat','lon'))
        rain            = nc_new.createVariable('precip','f4',('time','lat','lon'))
	date		= nc_new.createVariable('date',str,'time')

	# Creo una variable horaria para QVAPOR,QCLOUD,QRAIN
	for i in range(0,24):
		print(i)
		varQvapor 	= create_var(nc_new, i, "Qvapor")
		varQcloud	= create_var(nc_new, i, "Qcloud")
		varQrain	= create_var(nc_new, i, "Qrain")
		for j,d in enumerate(folders):
			files = os.listdir(parent_dir + d)
		        files.sort()

			#varQvapor[j,:,:,:]	= get_var_value(parent_dir,d,files[i],'QVAPOR')
			#varQcloud[j,:,:,:]	= get_var_value(parent_dir,d,files[i],'QCLOUD')
			#varQrain[j,:,:,:]	= get_var_value(parent_dir,d,files[i],'QRAIN')
			varQvapor[j,:,:,:], varQcloud[j,:,:,:], varQrain[j,:,:,:] = get_vars(parent_dir,d,files[i])
			print("."),

        # Datos a las variables
        latitude[:]     = lat
        longitude[:]    = lon
	height[:]	= hei

	# Obtenemos la lluvia acumulada para cada dia
	for i,d in enumerate(folders):
		print("Obteniendo datos de precipitacion para el dia...")
		print(d)
		rain[i,:,:]	= get_accumulated_rain(parent_dir,d)
		date[i]		= get_date(parent_dir,d)

        nc_new.close()

def create_var(nc_new, i, var):
	Q_v_var         = get_var_name(var,i)
        Q_v_var		= nc_new.createVariable(Q_v_var,'f4',('time','alt','lat','lon'))

	return Q_v_var

def get_vars(parent_dir,directory,f):
	nc      = netCDF4.Dataset(parent_dir + directory + "/" + f, 'r', format='NETCDF4')
	vapor	= nc.variables['QVAPOR'][:]
	cloud	= nc.variables['QCLOUD'][:]
	rain	= nc.variables['QRAIN'][:]

	return vapor, cloud, rain
	
def get_var_value(parent_dir,directory,f,var):
	nc	= netCDF4.Dataset(parent_dir + directory + "/" + f, 'r', format='NETCDF4')
        value 	= nc.variables[var][:]

	return value

def get_var_name(name,idx):
	
	var_name = name + str(idx)
	return var_name

def filter_nc(parent_dir):

	folders = os.listdir(parent_dir)
	folders.sort()
	print("Directorio del que vamos a obtener los datos...")
	print(parent_dir)

	for i in folders:
		print("Carpetas disponibles...")
		print(i)

	get_global_variables(parent_dir,folders)

	create_dset = 1

def main():

	parser = argparse.ArgumentParser(description = "Description for my parser")
	parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF", required = False, default = "")
	parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_* [Default: ncfiles]", required = False, default = "ncfiles")
	#parser.add_argument("-f", "--file", help = "Archino wrfout_*", required = False, default = "")

	argument = parser.parse_args()

	if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
	if argument.dir:
	    print("You have used '-D' or '--dir' with argument: {0}".format(argument.dir))
            filter_nc(format(argument.dir))

if __name__ == "__main__":
	main()


