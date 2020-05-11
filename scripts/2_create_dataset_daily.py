import os
import netCDF4
import pandas as pd
import numpy as np
import time as tm

import datetime as dt
import argparse

def create_csv(nc_filtrado,in_dir,out_dir,gt_file):

	filtrado        = netCDF4.Dataset(in_dir + "/" + nc_filtrado, 'r')
	verdad_terreno	= netCDF4.Dataset(gt_file, 'r')

	# Obtengo las variables del fichero filtrado (datos de prediccion)
        lat     	= filtrado.variables['XLAT'][:]
        lon     	= filtrado.variables['XLONG'][:]
	hei		= filtrado.variables['HGT'][:]
        rainc   	= filtrado.variables['RAINC'][:]
	rainnc  	= filtrado.variables['RAINNC'][:]
        time    	= filtrado.variables['DATE'][:]
	timestamp	= filtrado.variables['TIMESTAMP'][:]

	t_500	= filtrado.variables['T_500'][:]
	t_700   = filtrado.variables['T_700'][:]
	t_850   = filtrado.variables['T_850'][:]

	qvapor_500      = filtrado.variables['QVAPOR_500'][:]
        qvapor_700      = filtrado.variables['QVAPOR_700'][:]
        qvapor_850      = filtrado.variables['QVAPOR_850'][:]
        qcloud_500      = filtrado.variables['QCLOUD_500'][:]
        qcloud_700      = filtrado.variables['QCLOUD_700'][:]
        qcloud_850      = filtrado.variables['QCLOUD_850'][:]
        qrain_500       = filtrado.variables['QRAIN_500'][:]
        qrain_700       = filtrado.variables['QRAIN_700'][:]
        qrain_850       = filtrado.variables['QRAIN_850'][:]
        qice_500        = filtrado.variables['QICE_500'][:]
        qice_700        = filtrado.variables['QICE_700'][:]
        qice_850        = filtrado.variables['QICE_850'][:]
        qsnow_500       = filtrado.variables['QSNOW_500'][:]
        qsnow_700       = filtrado.variables['QSNOW_700'][:]
        qsnow_850       = filtrado.variables['QSNOW_850'][:]
        qgraup_500      = filtrado.variables['QGRAUP_500'][:]
        qgraup_700      = filtrado.variables['QGRAUP_700'][:]
        qgraup_850      = filtrado.variables['QGRAUP_850'][:]

        # Se define el numero de veces que vamos a repetir las coordenadas, que se va a repetir la fecha (fecha, x, y, z) y que se va a repetir todo
	repeat_date	= lat.shape[1]*lat.shape[2]

        # Obtenemos los valores de las variables en un array 1D
        lat_x   	= lat.flatten()
        lon_x   	= lon.flatten()
	hei_x		= hei.flatten()
        rainc_x  	= rainc.flatten()
	rainnc_x        = rainnc.flatten()
        time_x  	= time[:].flatten()
	timestamp_x	= timestamp[:].flatten()
	
	t_500_x		= t_500.flatten()
	t_700_x         = t_700.flatten()
	t_850_x         = t_850.flatten()

	qvapor_500_x    = qvapor_500.flatten()
        qvapor_700_x    = qvapor_700.flatten()
        qvapor_850_x    = qvapor_850.flatten()
        qcloud_500_x	= qcloud_500.flatten()
        qcloud_700_x    = qcloud_700.flatten()
        qcloud_850_x    = qcloud_850.flatten()
        qrain_500_x     = qrain_500.flatten()
        qrain_700_x     = qrain_700.flatten()
        qrain_850_x     = qrain_850.flatten()
        qice_500_x      = qice_500.flatten()
        qice_700_x      = qice_700.flatten()
        qice_850_x      = qice_850.flatten()
        qsnow_500_x     = qsnow_500.flatten()
        qsnow_700_x     = qsnow_700.flatten()
        qsnow_850_x     = qsnow_850.flatten()
        qgraup_500_x    = qgraup_500.flatten()
        qgraup_700_x    = qgraup_700.flatten()
        qgraup_850_x    = qgraup_850.flatten()


        # Creamos las series de cada una de las variables
        lat_serie       = pd.Series(lat_x,name='XLAT')
        lon_serie       = pd.Series(lon_x,name='XLONG')
	hei_serie       = pd.Series(hei_x,name='HGT')
        rainc_serie     = pd.Series(rainc_x,name='RAINC')
	rainnc_serie    = pd.Series(rainnc_x,name='RAINNC')
	t_500_serie	= pd.Series(t_500_x,name='T_500hPa')
	t_700_serie     = pd.Series(t_700_x,name='T_700hPa')
	t_850_serie     = pd.Series(t_850_x,name='T_850hPa')

	qvapor_500_serie	= pd.Series(qvapor_500_x,name='QVAPOR_500')
	qvapor_700_serie        = pd.Series(qvapor_700_x,name='QVAPOR_700')
	qvapor_850_serie        = pd.Series(qvapor_850_x,name='QVAPOR_850')
	qcloud_500_serie        = pd.Series(qcloud_500_x,name='QCLOUD_500')
	qcloud_700_serie        = pd.Series(qcloud_700_x,name='QCLOUD_700')
	qcloud_850_serie        = pd.Series(qcloud_850_x,name='QCLOUD_850')
	qrain_500_serie       	= pd.Series(qrain_500_x,name='QRAIN_500')
	qrain_700_serie         = pd.Series(qrain_700_x,name='QRAIN_700')
	qrain_850_serie         = pd.Series(qrain_850_x,name='QRAIN_850')
	qice_500_serie         	= pd.Series(qice_500_x,name='QICE_500')
	qice_700_serie          = pd.Series(qice_700_x,name='QICE_700')
	qice_850_serie          = pd.Series(qice_850_x,name='QICE_850')
	qsnow_500_serie         = pd.Series(qsnow_500_x,name='QSNOW_500')
	qsnow_700_serie         = pd.Series(qsnow_700_x,name='QSNOW_700')
	qsnow_850_serie         = pd.Series(qsnow_850_x,name='QSNOW_850')
	qgraup_500_serie        = pd.Series(qgraup_500_x,name='QGRAUP_500')
	qgraup_700_serie        = pd.Series(qgraup_700_x,name='QGRAUP_700')
	qgraup_850_serie        = pd.Series(qgraup_850_x,name='QGRAUP_850')

	# Obtenemos la verdad terreno y creamos una variable sintetica
	i_date		= dt.date(2008,1,1)
	print(int(nc_filtrado[5:7]))
	f_date		= dt.date(int(nc_filtrado[0:4]),int(nc_filtrado[5:7]),int(nc_filtrado[8:10]))
	#print(int(nc_filtrado[9:10]))
	inicio		= (f_date - i_date).days
	print(inicio)
        fin		= inicio + 1
        precip		= verdad_terreno.variables['precip'][inicio:fin,:,:]
	print(time.shape)
	print(lat.shape)
	print(precip.shape)
	
	repeat_precip	= 24
	precip_x	= precip.flatten()
	precip_serie	= pd.Series(precip_x,name='PRECIPITACION')
	precip_serie    = pd.concat([precip_serie]*repeat_precip, axis=0).reset_index(drop=True)
	
	sintetica_x     = np.where(precip_x>0, 1, 0)
	sintetica_serie	= pd.Series(sintetica_x,name='LLUVIA')
	sintetica_serie = pd.concat([sintetica_serie]*repeat_precip, axis=0).reset_index(drop=True)
	
	time_serie      = pd.Series(time_x,name='DATE').repeat(repeat_date).reset_index(drop=True)
	tmstamp_serie	= pd.Series(timestamp_x,name='TIMESTAMP').repeat(repeat_date).reset_index(drop=True)
	lat_serie	= pd.concat([lat_serie], axis=0).reset_index(drop=True)
	lon_serie	= pd.concat([lon_serie], axis=0).reset_index(drop=True)
	hei_serie	= pd.concat([hei_serie], axis=0).reset_index(drop=True)

	coordenadas	= pd.concat([lat_serie,lon_serie,hei_serie], axis=1)
	
	#final_serie	= pd.concat([time_serie,coordenadas,precip_serie,sintetica_serie], axis=1)
	final_serie     = pd.concat([time_serie,tmstamp_serie,coordenadas,rainc_serie,rainnc_serie,t_500_serie,t_700_serie,t_850_serie,qvapor_500_serie,qvapor_700_serie,qvapor_850_serie,qcloud_500_serie,qcloud_700_serie,qcloud_850_serie,qrain_500_serie,qrain_700_serie,qrain_850_serie,qice_500_serie,qice_700_serie,qice_850_serie,qsnow_500_serie,qsnow_700_serie,qsnow_850_serie,qgraup_500_serie,qgraup_700_serie,qgraup_850_serie,precip_serie,sintetica_serie], axis=1)
	#final_serie     = pd.concat([coordenadas,sintetica_serie], axis=1)


	########################

        # Concatenamos todas las series respecto al eje de las X (en horizontal)
	print("Creando final_serie...")

	# Eliminamos las filas para las que no tenemos los valores de la precip real
	print(final_serie.dropna(inplace=True))
	
	#remove		= nc_filtrado[0:10] + '[00:00:00]'
	#index		= final_serie.loc[final_serie['DATE']==remove].index.values.astype(int)
	index		= final_serie.loc[\
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
	final_serie	= final_serie.drop(index)
	
	# Almacenamos en un csv
	csv_name	= nc_filtrado[0:10]
        final_serie.to_csv(out_dir + "/" + csv_name + '.csv',index=False,header=True, float_format='%.8f')

def create_csv_files(in_dir,out_dir,gt_file):
	files = os.listdir(in_dir)
        files.sort()

        for i,nc_file in enumerate(files):
                print("Creando fichero .csv para el fichero...")
		print nc_file
		create_csv(nc_file,in_dir,out_dir,gt_file)
def main():

	parser = argparse.ArgumentParser(description = "Description for my parser")
	parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF", required = False, default = "")
	parser.add_argument("-D", "--dir", help = "Directorio con los archivos wrfout_d2_* [Default: ncfiles]", required = False, default = "ncfiles")
	parser.add_argument("-f", "--file", help = "Fichero con la verdad terreno [Default: none]", required = False, default = "outcsv")
	parser.add_argument("-O", "--output_dir", help = "Directorio para los .csv de salida [Default: out_csv]", required = False, default = "")

	argument = parser.parse_args()

	if argument.Help:
            print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
	if argument.output_dir:
            print("You have used '-O' or '--output_dir' with argument: {0}".format(argument.output_dir))
            out_dir = format(argument.output_dir)
            if not os.path.exists(format(argument.output_dir)):
                os.makedirs(format(argument.output_dir))
        if argument.dir:
            print("You have used '-D' or '--dir' with argument: {0}".format(argument.dir))
            #create_csv_files(format(argument.dir), out_dir)
        if argument.file :
            print("You have used '-f' or '--file' with argument: {0}".format(argument.file))
            create_csv_files(format(argument.dir), out_dir, format(argument.file))

if __name__ == "__main__":
	main()

