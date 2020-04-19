import os
import netCDF4
import pandas as pd
import numpy as np

import datetime as dt
import argparse
from optparse import OptionParser, Values

def get_variables(parent_dir,directory,nc_file):

	nc      = netCDF4.Dataset(parent_dir + "/" + directory + "/" + nc_file, 'r', format='NETCDF4')
	time	= nc.variables['XTIME']
	lat     = nc.variables['XLAT'][0,:,:]           
        lon     = nc.variables['XLONG'][0,:,:]          
        hei     = nc.variables['HGT'][0,:,:]            
	rainc	= nc.variables['RAINC'][0,:,:]
	rainnc  = nc.variables['RAINNC'][0,:,:]

	qvapor	= nc.variables['QVAPOR'][0,:,:,:]
	qcloud	= nc.variables['QCLOUD'][0,:,:,:]
	qrain	= nc.variables['QRAIN'][0,:,:,:]
	qice	= nc.variables['QICE'][0,:,:,:]
        qsnow   = nc.variables['QSNOW'][0,:,:,:]
        qgraup  = nc.variables['QGRAUP'][0,:,:,:]
	
	dtime   	= netCDF4.num2date(time[:],time.units)
	str_time        = [i.strftime("%Y-%m-%d[%H:%M:%S]") for i in dtime]

	print("Obteniendo fecha...")
	print(str_time)
	
	# Convertimos la lista a un string para que lo acepte netCDF
	str1 = " "     
	return str1.join(str_time),lat,lon,hei,rainc,rainnc,qvapor,qcloud,qrain,qice,qsnow,qgraup


def get_global_variables(parent_dir,folders,folder,out_dir):

	# Obtenemos las variables importantes
	files = os.listdir(parent_dir + "/" + folder)
	files.sort()

	print("Obteniendo valores de latitud y longitud del fichero...")
	f = parent_dir + "/" + folder + "/" + files[0]
	print(f)

        # Leemos el primer fichero nc dentro del directorio "ncfiles" (asumimos que todos son del mismo dominio)
        nc      = netCDF4.Dataset(f, 'r', format='NETCDF4')

        # Obtenemos las variables del fichero netcdf
        lat     = nc.variables['XLAT'][0,:,:]		# Latitud
        lon     = nc.variables['XLONG'][0,:,:]		# Longitud
	hei	= nc.variables['HGT'][0,:,:]		# Altitud
        time    = nc.variables['XTIME']			# Hora

        # Creamos las series de cada una de las variables
        dtime           = netCDF4.num2date(time[:],time.units)
        str_time        = [i.strftime("%Y-%m-%d") for i in dtime]

        # Creamos el fichero .nc de datos
	#Nota: quito el formato para poder utilizar strings en la fecha
        #nc_new  = netCDF4.Dataset(str_time[0] + '_filtered.nc','w', format='NETCDF4')
	print out_dir
	nc_new  = netCDF4.Dataset(out_dir + "/" + str_time[0] + '.nc','w')

        # Dimensiones
        dim_lat, dim_lon  	= lat.shape
	dim_time 		= 24
	dim_alt 		= 41
        print("Las dimensiones de la precipitacion son (time,lat,lon):")
        print(dim_time)
        print(dim_lat)
        print(dim_lon)
	print(dim_alt)
        print("--------------")

        # Configuramos nuestro nuevo ficher nc 
        # Definimos las dimensiones que tendran las distintas variables
        nc_new.createDimension('south_north',dim_lat)
        nc_new.createDimension('west_east',dim_lon)
        nc_new.createDimension('time',dim_time)
        nc_new.createDimension('bottom_top',dim_alt)

        latitude        = nc_new.createVariable('XLAT','f4',('time','south_north','west_east'))
        longitude       = nc_new.createVariable('XLONG','f4',('time','south_north','west_east'))
	height		= nc_new.createVariable('HGT','f4',('time','south_north','west_east'))
        rainc           = nc_new.createVariable('RAINC','f4',('time','south_north','west_east'))
	rainnc		= nc_new.createVariable('RAINNC','f4',('time','south_north','west_east'))
	date		= nc_new.createVariable('DATE',str,'time')

	Qvapor		= nc_new.createVariable('QVAPOR','f4',('time','bottom_top','south_north','west_east'))
	Qcloud		= nc_new.createVariable('QCLOUD','f4',('time','bottom_top','south_north','west_east'))
	Qrain		= nc_new.createVariable('QRAIN','f4',('time','bottom_top','south_north','west_east'))
        Qice	        = nc_new.createVariable('QICE','f4',('time','bottom_top','south_north','west_east'))
        Qsnow           = nc_new.createVariable('QSNOW','f4',('time','bottom_top','south_north','west_east'))
        Qgraup          = nc_new.createVariable('QGRAUP','f4',('time','bottom_top','south_north','west_east'))

	# Obtenemos todas las varialbles deseadas para cada dia
	files = os.listdir(parent_dir + "/" + folder)
	files.sort()
	
	for i,nc_file in enumerate(files):
		print("Obteniendo datos de precipitacion para el dia...")
		print(folder)
		print("Obteniendo datos de precipitacion para la hora...")
		print(nc_file)
		print(i)
		date[i],latitude[i,:,:],longitude[i,:,:],height[i,:,:],rainc[i,:,:],rainnc[i,:,:],Qvapor[i,:,:,:],Qcloud[i,:,:,:],Qrain[i,:,:,:],Qice[i,:,:,:],Qsnow[i,:,:,:],Qgraup[i,:,:,:]	= get_variables(parent_dir,folder,nc_file)

        nc_new.close()

def filter_nc(parent_dir, out_dir):

	folders = os.listdir(parent_dir)
	folders.sort()
	print("Directorio del que vamos a obtener los datos...")
	print(parent_dir)

	for i in folders:
		print("Carpetas disponibles...")
		print(i)
		get_global_variables(parent_dir,folders,i,out_dir)

def main():

	parser = argparse.ArgumentParser(description = "Description for my parser")
	parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF", required = False, default = "")
	parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_* [Default: ncfiles]", required = False, default = "ncfiles")
        parser.add_argument("-O", "--output_dir", help = "Directorio con los .nc generados* [Default: out_ncfiles]", required = False, default = "out_ncfiles")
	#parser.add_argument("-f", "--file", help = "Archino wrfout_*", required = False, default = "")

	argument = parser.parse_args()

	out_dir = ""

	if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
	if argument.output_dir:
            out_dir = format(argument.output_dir)
            if not os.path.exists(format(argument.output_dir)):
		os.makedirs(format(argument.output_dir))
	if argument.dir:
	    print("You have used '-D' or '--dir' with argument: {0}".format(argument.dir))
            filter_nc(format(argument.dir), out_dir)

if __name__ == "__main__":
	main()


