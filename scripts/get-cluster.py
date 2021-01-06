import argparse, os, joblib, fnmatch, json
import pandas as pd

STATS_PREFIX = "week"
SKIP_COLS_KEY = "global_skip_cols"
COLS_TO_ADD = ["gameId", "playId", "nflId"]
CLUSTER_KEY = "cluster"
PROB_KEY_PREFIX = "cluster_prob_"
ONLY_CLOSEST_KEY = "only_closest"
CLOSE_TO_BR_KEY = "close_to_br"

def get_cluster(gmm, config, data_folder, output_folder):
	stats_files = fnmatch.filter(os.listdir(data_folder), "{}*.csv".format(
		STATS_PREFIX))
	for f in stats_files:
		print("Processing stats file: {}".format(f))
		file_path = os.path.join(data_folder, f)
		data = pd.read_csv(file_path)
		if config[ONLY_CLOSEST_KEY] == 1:
			data = data.loc[data.groupby(GROUP_BY)[MAX_COL].idxmax()].reset_index(
				drop=True)
		elif len(config[CLOSE_TO_BR_KEY]) != 0:
			data = data[data[CLOSE_TO_BR_KEY].isin(config[CLOSE_TO_BR_KEY])]

		x = data.drop(config[SKIP_COLS_KEY], axis = 1)
		y = gmm.predict(x)
		y_prob = gmm.predict_proba(x)
		output_data = data[COLS_TO_ADD].copy()
		output_data[CLUSTER_KEY] = y
		y_split = list(zip(*y_prob))
		for i in range(len(y_split)):
			key = "{}{}".format(PROB_KEY_PREFIX, i)
			output_data[key] = y_split[i]
		output_file = os.path.join(output_folder, f)
		output_data.to_csv(output_file)
	print("Clustering output saved to {}".format(output_folder))

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the folder containing data files",
		required=True)
	parser.add_argument(
		"--config_path", type=str, help="specifies the path for config file",
		required=True)
	parser.add_argument(
		"--gmm_path", type=str, help="specifies the path for gmm joblib file",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	gmm_path = os.path.abspath(args["gmm_path"])
	output_path = os.path.abspath(args["output_path"])
	config_path = os.path.abspath(args["config_path"])
	with open(config_path) as f:
		config = json.load(f)
	print("Config: {}".format(config))
	gmm = joblib.load(gmm_path)

	get_cluster(gmm, config, data_path, output_path)

main()