import pandas as pd
import matplotlib.pyplot as plot
import os, argparse, json, math

X_FILE = "x.csv"
Y_PASS_FILE = "y-passResult.csv"
Y_DEFENSIVE_FILE = "y-isDefensivePI.csv"
Y_PENALTY_FILE = "y-penaltyOwner.csv"
Y_FILES = [Y_DEFENSIVE_FILE, Y_PASS_FILE, Y_PENALTY_FILE]
NORMALIZE_FILE = "normalizer.json"
IGNORE_X_KEY = "ignore_x_values"
IGNORE_Y_KEY = "ignore_y_values"

NORMALIZE_TYPES = {
	"playType": "playtypes",
	"passResult": "results",
	"penaltyOwner": "teamtypes",
	"typeDropback": "dropbacktypes",
	"offenseFormation": "formations"
}

def normalize(data, normalizer):
	replace_map = {}
	for col in data.columns:
		if col not in NORMALIZE_TYPES:
			continue
		norm_type = NORMALIZE_TYPES[col]
		col_map = normalizer[norm_type]
		flipped_map = dict(zip(col_map.values(), col_map.keys()))
		replace_map[col] = flipped_map
	data = data.replace(replace_map)

	return data

def plot_and_save_graph(data, output_folder, x_col, y_col, config):
	plot_data = data
	plot_indices = plot_data.index.values.tolist()
	found_indices = list(set(plot_indices) & set(config[IGNORE_X_KEY]))
	plot_data = plot_data.drop(found_indices, axis=0)
	plot_indices = plot_data.columns.values.tolist()
	found_indices = list(set(plot_indices) & set(config[IGNORE_Y_KEY]))
	plot_data = plot_data.drop(found_indices, axis = 1)

	axes = plot_data.plot.bar(stacked=True)
	plot.xticks(rotation=0, size=5)

	prefix = "{}-{}".format(x_col, y_col)
	output_path = os.path.join(output_folder, "{}.jpg".format(prefix))
	plot.savefig(output_path)
	output_path = os.path.join(output_folder, "{}.csv".format(prefix))
	data.to_csv(output_path)

def visualize(data_folder, output_folder, config_path):
	normalize_path = os.path.join(data_folder, NORMALIZE_FILE)
	with open(normalize_path, "r") as json_file:
		normalize_map = json.load(json_file)
	with open(config_path, "r") as json_file:
		config = json.load(json_file)

	x_path = os.path.join(data_folder, X_FILE)
	x_data = pd.read_csv(x_path)
	for y in Y_FILES:
		y_path = os.path.join(data_folder, y)
		y_data = pd.read_csv(y_path)
		for x_col in x_data.columns:
			for y_col in y_data.columns:
				data = x_data.join(y_data)
				data = data.groupby([x_col, y_col]).size().reset_index(
					name="count")
				data = normalize(data, normalize_map)
				pivot_data = data.pivot_table(values="count", index=x_col,
					columns=y_col)
				plot_and_save_graph(pivot_data, output_folder, x_col, y_col,
					config)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the folder containing data files",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	parser.add_argument(
		"--config_path", type=str, help="specifies the config file path",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])
	config_path = os.path.abspath(args["config_path"])

	visualize(data_path, output_path, config_path)
	print("Images and csv files saved to {}".format(output_path))

main()
