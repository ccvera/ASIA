import logging
logging.basicConfig(level=logging.INFO, filename='preprocess.log', format='%(asctime)s: %(levelname)s - %(message)s')
logger = logging.getLogger('RaP')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('RaP').addHandler(console)
logger = logging.getLogger(__name__)

logger.info('Started filtering variables')

import os
import netCDF4
import pandas as pd
import numpy as np
import time as tm

import datetime
import argparse
from optparse import OptionParser, Values
from wrf import getvar, interplevel

def set_T_value(v, hpa, var, name, parent_dir, folder):

	nc      = netCDF4.Dataset(parent_dir + "/" + folder + "/" + name, 'r', format='NETCDF4')
	hPa     = getvar(nc, "pressure")
	T	= getvar(nc, "temp", units="K")

	tem	= interplevel(T, hPa, hpa + '.')

	return tem[:,:]

	
def set_mixing_ratio_value(v, hpa, var, name, parent_dir, folder):

	nc	= netCDF4.Dataset(parent_dir + "/" + folder + "/" + name, 'r', format='NETCDF4')

	hPa     = getvar(nc, "pressure")
	x	= getvar(nc,v)

	mr	= interplevel(x, hPa, hpa + '.')

	return mr
	
def set_coord_value(var, name, parent_dir, folder):
	
	nc         = netCDF4.Dataset(parent_dir + "/" + folder + "/" + name, 'r', format='NETCDF4')
	return nc.variables[var][0,:,:]

def set_date_value(name, parent_dir, folder, timestamp=False):

	nc         = netCDF4.Dataset(parent_dir + "/" + folder + "/" + name, 'r', format='NETCDF4')

	time = nc.variables['XTIME']

        dtime           = netCDF4.num2date(time[:],time.units)
        str_time        = [i.strftime("%Y-%m-%d[%H:%M:%S]") for i in dtime]

        # Convertimos la lista a un string para que lo acepte netCDF
        str1 = " "

	str2 = ""
        d = datetime.datetime.strptime(str2.join(str_time), "%Y-%m-%d[%H:%M:%S]")
        tstamp = tm.mktime(d.timetuple())

	if timestamp is True:
		return tstamp
	else:
        	return str1.join(str_time)

# Leemos las variables de nuestro nc fuente	
def get_nc_var(nc, var):
	return getvar(nc, var)

def create_nc(parent_dir, folders, folder, out_dir):
	coordinates 	= ['XLAT', 'XLONG', 'HGT']
	mixing_ratio	= ['QVAPOR', 'QCLOUD', 'QRAIN', 'QICE', 'QSNOW', 'QGRAUP']
	precipitation 	= ['RAINC', 'RAINNC']

	new_vars 	= ['date', 'timestamp', 'latitude', 'longitude', 'height', 'rainc', 'rainnc', 'Qvapor_500', 'Qvapor_700', 'Qvapor_850', 'Qcloud_500', 'Qcloud_700', 'Qcloud_850', 'Qrain_500', 'Qrain_700', 'Qrain_850', 'Qice_500', 'Qice_700', 'Qice_850', 'Qsnow_500', 'Qsnow_700', 'Qsnow_850', 'Qgraup_500', 'Qgraup_700', 'Qgraup_850', 'T_500', 'T_700', 'T_850']
	new_vars_name	= ['DATE', 'TIMESTAMP', 'XLAT', 'XLONG', 'HGT', 'RAINC', 'RAINNC', 'QVAPOR_500', 'QVAPOR_700', 'QVAPOR_850', 'QCLOUD_500', 'QCLOUD_700', 'QCLOUD_850', 'QRAIN_500', 'QRAIN_700', 'QRAIN_850', 'QICE_500', 'QICE_700', 'QICE_850', 'QSNOW_500', 'QSNOW_700', 'QSNOW_850', 'QGRAUP_500', 'QGRAUP_700', 'QGRAUP_850', 'T_500', 'T_700', 'T_850']

	# Obtenemos las variables importantes
        files = os.listdir(parent_dir + "/" + folder)
        files.sort()

	# Creamos el nuevo nc
	f = parent_dir + "/" + folder + "/" + files[0]
	# Leemos el primer fichero nc dentro del directorio "ncfiles" (asumimos que todos son del mismo dominio)
        nc      = netCDF4.Dataset(f, 'r', format='NETCDF4')
	time    = nc.variables['XTIME']

        # Creamos las series de cada una de las variables
        dtime           = netCDF4.num2date(time[:],time.units)
        str_time        = [i.strftime("%Y-%m-%d") for i in dtime]

	nc_new  = netCDF4.Dataset(out_dir + "/" + str_time[0] + '.nc','w')
	
	# Dimensiones
	# Definimos las dimensiones que tendran las distintas variables
	dim_lat 	= 78
	dim_lon 	= 123
	dim_time	= 24
        nc_new.createDimension('south_north',dim_lat)
        nc_new.createDimension('west_east',dim_lon)
        nc_new.createDimension('time',dim_time)
	
	for i,v in enumerate(new_vars):
		if v == 'date':
			v = nc_new.createVariable(new_vars_name[i],str,'time')	
			for j in range(dim_time):
				v[j] = set_date_value(files[j],parent_dir,folder)
		elif v == 'timestamp':
			v = nc_new.createVariable(new_vars_name[i],'f4','time')
			for j in range(dim_time):
				v[j] = set_date_value(files[j],parent_dir,folder,timestamp=True)
		else:
			v = nc_new.createVariable(new_vars_name[i],'f4',('time','south_north','west_east'))
			for j in range(dim_time):
				if new_vars_name[i] in (coordinates, precipitation):
					v[j,:,:] = set_coord_value(new_vars_name[i],files[j],parent_dir,folder)
				else:
					var = new_vars_name[i]
					if var[:-4] in mixing_ratio:
						v[j,:,:] = set_mixing_ratio_value(var[:-4], var[-3:], new_vars_name[i],files[j],parent_dir,folder)
					if var[:-4] == 'T':
						v[j,:,:] = set_T_value(var[:-4], var[-3:], new_vars_name[i],files[j],parent_dir,folder)

	logger.info('File %s.nc created', str_time[0])
	nc_new.close()

def filter_nc(parent_dir, out_dir):
	folders = os.listdir(parent_dir)
	folders.sort()
	logger.info('RAW data directory: %s', parent_dir)

	# Para cada directorio diario se crea un .nc
	for i in folders:
		create_nc(parent_dir,folders,i,out_dir)


def main():
	# Parser para interepretar argumentos por linea de comandos
	parser = argparse.ArgumentParser()
	parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF")
	parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_* [Default: ncfiles]", required = False, default = "ncfiles")
	parser.add_argument("-O", "--output_dir", help = "Directorio con los .nc generados* [Default: out_ncfiles]", required = False, default = "out_ncfiles")
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
