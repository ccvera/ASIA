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

logger.info('Started creating_csv.py')

import os
import netCDF4
import math
import pandas as pd
import numpy as np
import time as tm

import datetime as dt
import argparse
from optparse import OptionParser, Values
from wrf import getvar, interplevel

def clean(final_serie,nc_filtrado):

	logger.info('Cleaning .csv file. Drop empty values and specific hours')
	final_serie.drop(final_serie.columns[0],axis=1,inplace=True)
	# Eliminamos las filas para las que no tenemos los valores de la precip real
        print(final_serie.dropna(inplace=True))
        index           = final_serie.loc[\
                        (final_serie['DATE']== nc_filtrado[0:10] + '[01:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[02:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[03:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[04:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[05:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[06:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[07:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[08:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[09:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[10:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[11:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[12:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[13:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[14:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[15:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[16:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[17:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[18:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[19:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[20:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[21:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[22:00:00]') | \
                        (final_serie['DATE']== nc_filtrado[0:10] + '[23:00:00]')].index.values.astype(int)
	return final_serie.drop(index)
	#return final_serie


def get_conditions(var):
	logger.debug('Get conditions to create "RANGO"')
        conditions      = [var<=0.1,\
                        (var>0.1)&(var<=1),\
                        (var>1)&(var<=5),\
                        (var>5)&(var<=20),\
                        (var>20)&(var<=40),\
                        (var>40)&(var<=80),\
                        (var>80)]

        choices         = [ 0, 1, 2, 3, 4, 5, 6 ]

	return conditions,choices
'''
def get_conditions(var):
        logger.debug('Get conditions to create "RANGO"')
        conditions      = [var<0.1,\
                        (var>=0.1)&(var<2.5),\
                        (var>=2.5)&(var<5),\
                        (var>=5)]

        choices         = [ 0, 1, 2, 3 ]

        return conditions,choices
'''

def create_csv(nc_filtrado,in_dir,out_dir,gt_file):

	logger.info('Creating .csv file')

        filtrado        = netCDF4.Dataset(in_dir + "/" + nc_filtrado, 'r')
        verdad_terreno  = netCDF4.Dataset(gt_file, 'r')

	nc_vars_name	= ['DATE', 'TIMESTAMP', 'XLAT', 'XLONG', 'HGT', 'RAINC', 'RAINNC', 'QVAPOR_500', 'QVAPOR_700', 'QVAPOR_850', 'QCLOUD_500', 'QCLOUD_700', 'QCLOUD_850', 'QRAIN_500', 'QRAIN_700', 'QRAIN_850', 'QICE_500', 'QICE_700', 'QICE_850', 'QSNOW_500', 'QSNOW_700', 'QSNOW_850', 'QGRAUP_500', 'QGRAUP_700', 'QGRAUP_850', 'T_500', 'T_700', 'T_850']
	pd_vars_name	= ['time', 'timestamp', 'lat', 'lon', 'hei', 'rainc', 'rainnc', 'qvapor_500', 'qvapor_700', 'qvapor_850', 'qcloud_500', 'qcloud_700', 'qcloud_850', 'qrain_500', 'qrain_700', 'qrain_850', 'qice_500', 'qice_700', 'qice_850', 'qsnow_500', 'qsnow_700', 'qsnow_850', 'qgraup_500', 'qgraup_700', 'qgraup_850', 't_500', 't_700', 't_850']
	date		= ['time', 'timestamp']
	coords		= ['time', 'timestamp', 'lat', 'lon', 'hei', 'rainc', 'rainnc']
	variables	= ['qvapor_500', 'qvapor_700', 'qvapor_850', 'qcloud_500', 'qcloud_700', 'qcloud_850', 'qrain_500', 'qrain_700', 'qrain_850', 'qice_500', 'qice_700', 'qice_850', 'qsnow_500', 'qsnow_700', 'qsnow_850', 'qgraup_500', 'qgraup_700', 'qgraup_850', 't_500', 't_700', 't_850']
	rain		= ['rainc', 'rainnc']

	final = pd.Series() 	
	lat = filtrado.variables['XLAT'][:]
	repeat_date     = lat.shape[1]*lat.shape[2]
	
	'''
	for i,v in enumerate(pd_vars_name):
		v 	= filtrado.variables[nc_vars_name[i]][:]
		v_x 	= pd_vars_name[i] + '_x'
		v_x 	= v.flatten()
		v_serie	= pd_vars_name[i] + '_serie'
		v_serie	= pd.Series(v_x,name=nc_vars_name[i])
		# Agnadimos las variables procesadas
		if pd_vars_name[i] in date:
			v_serie = pd.concat([v_serie], axis=0).repeat(repeat_date).reset_index(drop=True)	
		final	= pd.concat([final,v_serie], axis=1)
	'''

	for i,v in enumerate(pd_vars_name):
		for j in range(1,25):
			if j%3==0: 
				if pd_vars_name[i] in variables:
					v       = filtrado.variables[nc_vars_name[i]][j-1,:]
					v_x     = pd_vars_name[i] + '_x_' + str(j)
			                v_x     = v.flatten()
			                v_serie = pd_vars_name[i] + '_serie_' + str(j)
					v_name	= nc_vars_name[i] + '_' + str(j) + 'h'
			                v_serie = pd.Series(v_x,name=v_name)
					
					final   = pd.concat([final,v_serie], axis=1)
			if j==3:
				if pd_vars_name[i] not in variables:
					# Cojo el 24 para coger la acumulada de la precipitacion y las coordenadas se mantienen para todas las horas
					if pd_vars_name[i] in date:
						v       = filtrado.variables[nc_vars_name[i]][23]
						v_x     = pd_vars_name[i] + '_x'
						v_x	= v
						v_serie = pd_vars_name[i] + '_serie'
						v_serie = pd.Series(v_x,name=nc_vars_name[i])
	
						v_serie = pd.concat([v_serie], axis=0).repeat(repeat_date).reset_index(drop=True)

					else:
						v       = filtrado.variables[nc_vars_name[i]][23,:]
				                v_x     = pd_vars_name[i] + '_x'
				                v_x     = v.flatten()
				                v_serie = pd_vars_name[i] + '_serie'
				                v_serie = pd.Series(v_x,name=nc_vars_name[i])
	
					final   = pd.concat([final,v_serie], axis=1)


	logger.info('All variables obteined')

	# Obtenemos el rango de fechas que queremos de la verdad terreno
	i_date          = dt.date(2008,1,1)
        f_date          = dt.date(int(nc_filtrado[0:4]),int(nc_filtrado[5:7]),int(nc_filtrado[8:10]))
	inicio          = (f_date - i_date).days
	fin             = inicio + 1
	
	precip          = verdad_terreno.variables['precip'][inicio:fin,:,:]
	
	# Normalizamos la fecha para obtener el agno y el mes al que pertenecen los datos
        rep             = final.shape[0]
        year            = int(nc_filtrado[0:4])
        month           = int(nc_filtrado[5:7])
        
	year_serie      = pd.Series(year).repeat(rep).reset_index(drop=True)
        month_serie     = pd.Series(month).repeat(rep).reset_index(drop=True)
        year_serie      = year_serie.rename("YEAR")
        month_serie     = month_serie.rename("MONTH")

        final = pd.concat([final,year_serie,month_serie], axis=1)

	# Obtenemos la precipitacion para el WRF
	precip_wrf_serie = final['RAINC'].add(final['RAINNC'])
	precip_wrf_serie = precip_wrf_serie.rename("PRECIPITACION_WRF")

	# Obtenemos la precipitacion para la CHE
	repeat_precip   = 24
        precip_x        = precip.flatten()
        precip_serie    = pd.Series(precip_x,name='PRECIPITACION')
        precip_serie    = pd.concat([precip_serie]*repeat_precip, axis=0).reset_index(drop=True)
	
	# Obtenemos la binaria para el WRF
	sintetica_wrf_x     = np.where(precip_wrf_serie>0, 1, 0)
        sintetica_wrf_serie = pd.Series(sintetica_wrf_x,name='LLUVIA_WRF')
        sintetica_wrf_serie = pd.concat([sintetica_wrf_serie]*repeat_precip, axis=0).reset_index(drop=True)

	# Obtenemos la binaria para la CHE
	sintetica_x     = np.where(precip_serie>0, 1, 0)
        sintetica_serie = pd.Series(sintetica_x,name='LLUVIA')
        sintetica_serie = pd.concat([sintetica_serie]*repeat_precip, axis=0).reset_index(drop=True)

	# Obtenemos los rangos para el WRF
	conditions,choices  = get_conditions(precip_wrf_serie)
	rango_wrf_x         = np.select(conditions, choices, default='zero')
        rango_wrf_serie     = pd.Series(rango_wrf_x,name='RANGO_WRF')
        rango_wrf_serie     = pd.concat([rango_wrf_serie]*repeat_precip, axis=0).reset_index(drop=True)

	# Obtenemos los rangos para la CHE
	conditions,choices	= get_conditions(precip_x)
	rango_x         	= np.select(conditions, choices, default='zero')
        rango_serie     	= pd.Series(rango_x,name='RANGO')
        rango_serie     	= pd.concat([rango_serie]*repeat_precip, axis=0).reset_index(drop=True)

	# Agnadimos las variables procesadas
	final = pd.concat([final,precip_wrf_serie,precip_serie,sintetica_wrf_serie,sintetica_serie,rango_wrf_serie,rango_serie], axis=1)
	#final = pd.concat([year_serie,month_serie,precip_wrf_serie], axis=1)
		
        # ECM para la binaria
        ecm_sintetica_serie     = final['LLUVIA'].sub(final['LLUVIA_WRF'])
	#ecm_sintetica_serie	= ecm_sintetica_serie.mul(ecm_sintetica_serie)
        ecm_sintetica_serie     = ecm_sintetica_serie.rename("MSE_LLUVIA")

        # ECM para el rango
        ecm_rango_serie     = final['RANGO'].fillna(0).astype(int).sub(final['RANGO_WRF'].fillna(0).astype(int))
        #ecm_rango_serie     = ecm_rango_serie.mul(ecm_rango_serie)
        ecm_rango_serie     = ecm_rango_serie.rename("MSE_RANGO")

	# Agnadimos el mse
	final = pd.concat([final,ecm_sintetica_serie,ecm_rango_serie], axis=1)

	'''
	# Normalizamos la fecha para obtener el agno y el mes al que pertenecen los datos
        rep             = final.shape[0]
        year            = int(nc_filtrado[0:4])
        month           = int(nc_filtrado[5:7])
        print year
        print month
        year_serie      = pd.Series(year).repeat(rep).reset_index(drop=True)
        month_serie     = pd.Series(month).repeat(rep).reset_index(drop=True)
        year_serie      = year_serie.rename("YEAR")
        month_serie     = month_serie.rename("MONTH")

        final = pd.concat([year_serie,month_serie,final], axis=1)
	'''

	# Limpiamos el dataframe
	final = clean(final,nc_filtrado)
	
	# Almacenamos en un csv
        csv_name        = nc_filtrado[0:10]
	final.to_csv(out_dir + "/" + csv_name + '.csv',index=False,header=True, float_format='%.8f')
	logger.info('%s.csv created', csv_name)


def create_csv_files(in_dir,out_dir,gt_file):
        files = os.listdir(in_dir)
        files.sort()

        for i,nc_file in enumerate(files):
                create_csv(nc_file,in_dir,out_dir,gt_file)
	
def main():

        parser = argparse.ArgumentParser(description = "Description for my parser")
        parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF")
        parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_*")
        parser.add_argument("-f", "--file", help = "Fichero con la verdad terreno [Default: none]")
        parser.add_argument("-O", "--output_dir", help = "Directorio para los .csv de salida [Default: out_csv]")

        argument = parser.parse_args()

        if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
        if argument.output_dir:
            out_dir = format(argument.output_dir)
            if not os.path.exists(format(argument.output_dir)):
                os.makedirs(format(argument.output_dir))
        if argument.file :
            create_csv_files(format(argument.dir), out_dir, format(argument.file))

if __name__ == "__main__":
        main()
