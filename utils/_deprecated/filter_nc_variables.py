import os
import netCDF4
import pandas as pd
import numpy as np
import time as tm

import datetime
import argparse
from optparse import OptionParser, Values
from wrf import getvar, interplevel

def get_temperature(nc):

	T	= getvar(nc, "temp", units="K")
	hPa	= getvar(nc, "pressure")

	T_500	= interplevel(T, hPa, 500.)
	T_700   = interplevel(T, hPa, 700.)
	T_850   = interplevel(T, hPa, 850.)

	return T_500[:,:],T_700[:,:],T_850[:,:]

def get_razon_de_mezcla(nc):

	hPa     = getvar(nc, "pressure")

	qvapor  = getvar(nc,'QVAPOR')
        qcloud  = getvar(nc,'QCLOUD')
        qrain   = getvar(nc,'QRAIN')
        qice    = getvar(nc,'QICE')
        qsnow   = getvar(nc,'QSNOW')
        qgraup  = getvar(nc,'QGRAUP')

	qvapor_500	= interplevel(qvapor, hPa, 500.)
	qvapor_700      = interplevel(qvapor, hPa, 700.)
	qvapor_850      = interplevel(qvapor, hPa, 850.)
	qcloud_500      = interplevel(qcloud, hPa, 500.)
	qcloud_700      = interplevel(qcloud, hPa, 700.)
	qcloud_850      = interplevel(qcloud, hPa, 850.)
	qrain_500      	= interplevel(qrain, hPa, 500.)
	qrain_700      	= interplevel(qrain, hPa, 700.)
	qrain_850      	= interplevel(qrain, hPa, 850.)
	qice_500	= interplevel(qice, hPa, 500.)
	qice_700        = interplevel(qice, hPa, 700.)
	qice_850        = interplevel(qice, hPa, 850.)
	qsnow_500       = interplevel(qsnow, hPa, 500.)
	qsnow_700       = interplevel(qsnow, hPa, 700.)
	qsnow_850       = interplevel(qsnow, hPa, 850.)
	qgraup_500      = interplevel(qgraup, hPa, 500.)
	qgraup_700      = interplevel(qgraup, hPa, 700.)
	qgraup_850      = interplevel(qgraup, hPa, 850.)

	return qvapor_500[:,:],qvapor_700[:,:],qvapor_850[:,:],qcloud_500[:,:],qcloud_700[:,:],qcloud_850[:,:],qrain_500[:,:],qrain_700[:,:],qrain_850[:,:],qice_500[:,:],qice_700[:,:],qice_850[:,:],qsnow_500[:,:],qsnow_700[:,:],qsnow_850[:,:],qgraup_500[:,:],qgraup_700[:,:],qgraup_850[:,:]

def get_variables(nc):

	time	= nc.variables['XTIME']
	lat     = nc.variables['XLAT'][0,:,:]           
        lon     = nc.variables['XLONG'][0,:,:]          
        hei     = nc.variables['HGT'][0,:,:]            
	rainc	= nc.variables['RAINC'][0,:,:]
	rainnc  = nc.variables['RAINNC'][0,:,:]

	dtime   	= netCDF4.num2date(time[:],time.units)
	str_time        = [i.strftime("%Y-%m-%d[%H:%M:%S]") for i in dtime]

	#timestamp	= tm.mktime(tm.strptime([i.strftime("%Y-%m-%s %H:%M:%S") for i in dtime], "%Y-%m-%d %H:%M:%S"))

	date_string = "11/22/2019"
	str2 = ""
	d = datetime.datetime.strptime(str2.join(str_time), "%Y-%m-%d[%H:%M:%S]")
	#timestamp = datetime.datetime.timestamp(d)
	timestamp = tm.mktime(d.timetuple())
	#print(timestamp)

	print("Obteniendo fecha...")
	print(str_time)
	print(timestamp)
	
	# Convertimos la lista a un string para que lo acepte netCDF
	str1 = " "     
	return str1.join(str_time),timestamp,lat,lon,hei,rainc,rainnc


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
	#fol 			= os.listdir(parent_dir+"/"+folder)
	fol			= len([name for name in os.listdir(parent_dir+"/"+folder) if os.path.isfile(name)])
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

        latitude        = nc_new.createVariable('XLAT','f4',('time','south_north','west_east'))
        longitude       = nc_new.createVariable('XLONG','f4',('time','south_north','west_east'))
	height		= nc_new.createVariable('HGT','f4',('time','south_north','west_east'))
        rainc           = nc_new.createVariable('RAINC','f4',('time','south_north','west_east'))
	rainnc		= nc_new.createVariable('RAINNC','f4',('time','south_north','west_east'))
	date		= nc_new.createVariable('DATE',str,'time')
	timestamp	= nc_new.createVariable('TIMESTAMP','f4','time')

	Qvapor_500	= nc_new.createVariable('QVAPOR_500','f4',('time','south_north','west_east'))
	Qvapor_700      = nc_new.createVariable('QVAPOR_700','f4',('time','south_north','west_east'))
	Qvapor_850      = nc_new.createVariable('QVAPOR_850','f4',('time','south_north','west_east'))
	Qcloud_500      = nc_new.createVariable('QCLOUD_500','f4',('time','south_north','west_east'))
	Qcloud_700      = nc_new.createVariable('QCLOUD_700','f4',('time','south_north','west_east'))
	Qcloud_850      = nc_new.createVariable('QCLOUD_850','f4',('time','south_north','west_east'))
	Qrain_500      	= nc_new.createVariable('QRAIN_500','f4',('time','south_north','west_east'))
	Qrain_700      	= nc_new.createVariable('QRAIN_700','f4',('time','south_north','west_east'))
	Qrain_850      	= nc_new.createVariable('QRAIN_850','f4',('time','south_north','west_east'))
	Qice_500	= nc_new.createVariable('QICE_500','f4',('time','south_north','west_east'))
	Qice_700        = nc_new.createVariable('QICE_700','f4',('time','south_north','west_east'))
	Qice_850        = nc_new.createVariable('QICE_850','f4',('time','south_north','west_east'))
	Qsnow_500       = nc_new.createVariable('QSNOW_500','f4',('time','south_north','west_east'))
	Qsnow_700       = nc_new.createVariable('QSNOW_700','f4',('time','south_north','west_east'))
	Qsnow_850       = nc_new.createVariable('QSNOW_850','f4',('time','south_north','west_east'))
	Qgraup_500      = nc_new.createVariable('QGRAUP_500','f4',('time','south_north','west_east'))
	Qgraup_700      = nc_new.createVariable('QGRAUP_700','f4',('time','south_north','west_east'))
	Qgraup_850      = nc_new.createVariable('QGRAUP_850','f4',('time','south_north','west_east'))

	T_500		= nc_new.createVariable('T_500','f4',('time','south_north','west_east'))
	T_700		= nc_new.createVariable('T_700','f4',('time','south_north','west_east'))
	T_850           = nc_new.createVariable('T_850','f4',('time','south_north','west_east'))

	# Obtenemos todas las varialbles deseadas para cada dia
	files = os.listdir(parent_dir + "/" + folder)
	files.sort()
	
	for i,nc_file in enumerate(files):
		print("Obteniendo datos de precipitacion para el dia...")
		print(folder)
		print("Obteniendo datos de precipitacion para la hora...")
		print(nc_file)
		nc_file 	= netCDF4.Dataset(parent_dir + "/" + folder + "/" + nc_file, 'r', format='NETCDF4')
		print(i)
		date[i],timestamp[i],latitude[i,:,:],longitude[i,:,:],height[i,:,:],rainc[i,:,:],rainnc[i,:,:] = get_variables(nc_file)
		T_500[i,:,:],T_700[i,:,:],T_850[i,:,:] = get_temperature(nc_file)
		Qvapor_500[i,:,:],Qvapor_700[i,:,:],Qvapor_850[i,:,:],Qcloud_500[i,:,:],Qcloud_700[i,:,:],Qcloud_850[i,:,:],Qrain_500[i,:,:],Qrain_700[i,:,:],Qrain_850[i,:,:],Qice_500[i,:,:],Qice_700[i,:,:],Qice_850[i,:,:],Qsnow_500[i,:,:],Qsnow_700[i,:,:],Qsnow_850[i,:,:],Qgraup_500[i,:,:],Qgraup_700[i,:,:],Qgraup_850[i,:,:] = get_razon_de_mezcla(nc_file)

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


