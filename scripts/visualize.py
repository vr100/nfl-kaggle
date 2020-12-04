import pandas as pd
import matplotlib.pyplot as plot
import os, argparse, json, math

X_FILE = "x.csv"
Y_PASS_FILE = "y-passResult.csv"
Y_DEFENSIVE_FILE = "y-isDefensivePI.csv"
Y_PENALTY_FILE = "y-penaltyOwner.csv"
Y_FILES = [Y_DEFENSIVE_FILE, Y_PASS_FILE, Y_PENALTY_FILE]
NORMALIZE_FILE = "normalizer.json"
DPI = 100

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

def plot_and_save_graph(data, output_folder, x_col, y_col):
	axes = data.plot.bar(stacked=True)
	plot.xticks(rotation=0)

	prefix = "{}-{}".format(x_col, y_col)
	output_path = os.path.join(output_folder, "{}.jpg".format(prefix))
	plot.savefig(output_path)
	output_path = os.path.join(output_folder, "{}.csv".format(prefix))
	data.to_csv(output_path)

def visualize(data_folder, output_folder):
	normalize_path = os.path.join(data_folder, NORMALIZE_FILE)
	with open(normalize_path, "r") as json_file:
		normalize_map = json.load(json_file)

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
				plot_and_save_graph(pivot_data, output_folder, x_col, y_col)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the folder containing data files",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])

	visualize(data_path, output_path)
	print("Images and csv files saved to {}".format(output_path))

main()
