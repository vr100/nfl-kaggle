import argparse, os, fnmatch
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# Reference paper - https://arxiv.org/abs/1906.11373
# "Unsupervised Methods for Identifying Pass Coverage Among Defensive Backs with NFL Player Tracking Data"


STATS_PREFIX = "week"
GROUP_COUNT = 5

SKIP_COLS = ["gameId", "playId", "nflId"]

def run_gmm_for_group_count(file_data, group_count):
	print("Running gmm for group count {}".format(group_count))
	ari = []
	gmm = []
	file_count = len(file_data)
	for k in range(file_count):
		#print("Running gmm for {}".format(k))
		data = pd.DataFrame()
		for j in range(file_count):
			if j == k:
				continue
			data = data.append(file_data[j], ignore_index=True)

		x_without_k = data.drop(SKIP_COLS, axis = 1).dropna()
		gmm_without_k = GaussianMixture(n_components=group_count,
			covariance_type="full", max_iter=1000)
		gmm_without_k = gmm_without_k.fit(x_without_k)

		x_k = file_data[k].drop(SKIP_COLS, axis = 1).dropna()
		gmm_k = GaussianMixture(n_components=group_count,
			covariance_type="full", max_iter=1000)
		gmm_k = gmm_k.fit(x_k)

		# predict cluster for the k week on both models
		# gmm without k data, gmm with k data
		y_k_on_without_k_model = gmm_without_k.predict(x_k)
		y_k = gmm_k.predict(x_k)

		ari_k = adjusted_rand_score(y_k_on_without_k_model, y_k)
		ari.append(ari_k)
		gmm.append(gmm_without_k)

	ari_max_index = ari.index(max(ari))
	ari_max = ari[ari_max_index]
	gmm_max = gmm[ari_max_index]
	ari_sum = sum(ari)
	result = {
		"max_ari": ari_max,
		"total_ari": ari_sum,
		"gmm": gmm_max
	}
	return result

def print_results(gmm_groups, max_index):
	groups = sorted(gmm_groups.keys())
	print("Results: ")
	for g in groups:
		print("{}: Max: {}, Total: {}".format(g, gmm_groups[g]["max_ari"],
			gmm_groups[g]["total_ari"]))
	print("Selected group count: {}, Max ARI: {}".format(max_index,
		gmm_groups[max_index]["max_ari"]))

def run_gmm(data_folder):
	stats_files = fnmatch.filter(os.listdir(data_folder), "{}*.csv".format(
		STATS_PREFIX))
	file_data = []
	for sf in stats_files:
		print("Working on file {} ...".format(sf))
		input_file = os.path.join(data_folder, sf)
		stats_data = pd.read_csv(input_file)
		file_data.append(stats_data)

	gmm_groups = {}
	for g in range(2, 10):
		result = run_gmm_for_group_count(file_data, g)
		gmm_groups[g] = result

	max_index = max(gmm_groups, key= lambda x: gmm_groups[x]["total_ari"])
	print_results(gmm_groups, max_index)

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
