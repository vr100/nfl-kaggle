import argparse, os, fnmatch
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# Reference paper - https://arxiv.org/abs/1906.11373
# "Unsupervised Methods for Identifying Pass Coverage Among Defensive Backs with NFL Player Tracking Data"


STATS_PREFIX = "week"
GROUP_COUNT = 5

GLOBAL_SKIP_COLS = ["gameId", "playId", "nflId", "Unnamed: 0"]

def run_gmm_for_g_and_k(file_data, g, k, skip_cols = []):
	skip_cols = skip_cols + GLOBAL_SKIP_COLS
	file_count = len(file_data)
	data = pd.DataFrame()
	for j in range(file_count):
		if j == k:
			continue
		data = data.append(file_data[j], ignore_index=True)

	x = data.drop(skip_cols, axis = 1).dropna()
	gmm = GaussianMixture(n_components=g,
		covariance_type="full", max_iter=1000)
	gmm = gmm.fit(x)

	x_k = file_data[k].drop(skip_cols, axis = 1).dropna()
	gmm_k = GaussianMixture(n_components=g,
		covariance_type="full", max_iter=1000)
	gmm_k = gmm_k.fit(x_k)

	# predict cluster for the k week on both models
	y = gmm.predict(x_k)
	y_k = gmm_k.predict(x_k)

	ari = adjusted_rand_score(y, y_k)
	# return the computed ari and gmm (skipping k)
	return (ari, gmm)

def run_gmm_for_group_count(file_data, group_count):
	print("Running gmm for group count {}".format(group_count))
	ari = []
	gmm = []
	file_count = len(file_data)
	for k in range(file_count):
		# print("Running gmm by leaving out index {}".format(k))
		(ari_k, gmm_k) = run_gmm_for_g_and_k(file_data, group_count, k)
		ari.append(ari_k)
		gmm.append(gmm_k)

	ari_max_index = ari.index(max(ari))
	ari_max = ari[ari_max_index]
	gmm_max = gmm[ari_max_index]
	ari_sum = sum(ari)
	result = {
		"lowo_index": ari_max_index,
		"max_ari": ari_max,
		"total_ari": ari_sum,
		"gmm": gmm_max
	}
	return result

def run_gmm_feature_influence(file_data, group_count, skip_lowo):
	print("Running gmm for group {}, skipping lowo index: {}".format(
		group_count, skip_lowo))
	if len(file_data) == 0:
		return
	cols = set(file_data[0].columns) - set(GLOBAL_SKIP_COLS)
	result = {}
	for c in cols:
		print("Skipping feature {}".format(c))
		ari_c, gmm_c = run_gmm_for_g_and_k(file_data, group_count, skip_lowo,
			skip_cols = [c])
		result[c] = {
			"ari": ari_c,
			"gmm": gmm_c
		}
	return result

def print_feature_influence_results(result, ari_with_all):
	influence = {}
	for feature in result:
		influence[feature] = ari_with_all - result[feature]["ari"]
	influence =  dict(sorted(influence.items(), key=lambda item: item[1],
		reverse=True))
	print("Results: (ari with all features: {})".format(ari_with_all))
	for feature in influence:
		print("{}: Influence: {}, ari: {}".format(feature, influence[feature],
			result[feature]["ari"]))

def print_group_results(gmm_groups, selected):
	groups = sorted(gmm_groups.keys())
	print("Results: ")
	for g in groups:
		print("{}: Max: {}, Total: {}".format(g, gmm_groups[g]["max_ari"],
			gmm_groups[g]["total_ari"]))
	print("Selected group count: {}, Max ARI: {}".format(selected,
		gmm_groups[selected]["max_ari"]))

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

	selected_group = max(gmm_groups, key= lambda x: gmm_groups[x]["total_ari"])
	gmm_influence_result = run_gmm_feature_influence(file_data, selected_group,
		gmm_groups[selected_group]["lowo_index"])

	print_group_results(gmm_groups, selected_group)
	print_feature_influence_results(gmm_influence_result,
		gmm_groups[selected_group]["max_ari"])

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
