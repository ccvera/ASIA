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

def create_nc_var():
	pass	

# Leemos las variables de nuestro nc fuente	
def get_nc_var(nc, var):
	return getvar(nc, var)

def create_nc(parent_dir, folders, folder, out_dir):
	coordinates 	= ['XLAT', 'XLONG', 'HGT', 'XTIME']
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
		elif v == 'timestamp':
			v = nc_new.createVariable(new_vars_name[i],'f4','time')
		else:
			v = nc_new.createVariable(new_vars_name[i],'f4',('time','south_north','west_east'))

	for j,nc_file in enumerate(files):
		nc_file         = netCDF4.Dataset(parent_dir + "/" + folder + "/" + nc_file, 'r', format='NETCDF4')
		        for i,v in enumerate(new_vars):
		                if v == 'date':
					print j
		                elif v == 'timestamp':
					print j
		                else:
					print j

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
