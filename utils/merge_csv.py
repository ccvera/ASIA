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

logger.info('Started creating merged csv...')

import os, glob
import numpy as np
import pandas as pd

def merge_csv(path, csv_name):
	all_files	= glob.glob(os.path.join(path, "*.csv"))

	df_merged       = pd.concat(pd.read_csv(f) for f in all_files)
	df_merged.to_csv(csv_name, index=False,header=True, float_format='%.8f')

	logger.info('%s created', csv_name)

def main():
	# Parser para interepretar argumentos por linea de comandos
	parser = argparse.ArgumentParser()
	parser.add_argument("-H", "--Help", help = "Script para la creacion de .csv diarios a partir de las salidas horarias de WRF")
	parser.add_argument("-t", "--dir_train", help = "Directorio con los archivos .csv del dataset de train", required = False)
	parser.add_argument("-v", "--dir_val", help = "Directorio con los .csv del dataset de validation", required = False)
	argument = parser.parse_args()

	out_dir = ""

	if argument.Help:
		print("You have used '-H' or '--Help' with argument: {0}".format(argument.Help))
	if argument.dir_train:
		merge_csv(format(argument.dir_train), "merged_train.csv")
	if argument.dir_val:
		merge_csv(format(argument.dir_val), "merged_validation.csv")
	

if __name__ == "__main__":
	main()
