import argparse, os, fnmatch
import pandas as pd
from sklearn.mixture import GaussianMixture

STATS_PREFIX = "week"
GROUP_COUNT = 5

SKIP_COLS = ["gameId", "playId", "nflId"]

def run_gmm(data_folder):
	stats_files = fnmatch.filter(os.listdir(data_folder), "{}*.csv".format(
		STATS_PREFIX))
	data = pd.DataFrame()
	for sf in stats_files:
		print("Working on file {} ...".format(sf))
		input_file = os.path.join(data_folder, sf)
		stats_data = pd.read_csv(input_file)
		data = data.append(stats_data, ignore_index=True)
	X = data.drop(SKIP_COLS, axis = 1).dropna()
	### using covariance_type = "full" (that is, VVV covariance structure)
	gmm = GaussianMixture(n_components=GROUP_COUNT, covariance_type="full")
	gmm = gmm.fit(X)
	print(gmm.means_, gmm.covariances_)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the folder containing data files",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])

	run_gmm(data_path)

main()
