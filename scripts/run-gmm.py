import argparse, os, fnmatch, json, joblib
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

# Reference paper - https://arxiv.org/abs/1906.11373
# "Unsupervised Methods for Identifying Pass Coverage Among Defensive Backs with NFL Player Tracking Data"

STATS_PREFIX = "week"
SKIP_COLS_KEY = "global_skip_cols"

def run_gmm_for_g_and_k(file_data, g, k, skip_cols):
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

def run_gmm_for_group_count(file_data, group_count, config):
	print("Running gmm for group count {}".format(group_count))
	ari = []
	gmm = []
	file_count = len(file_data)
	for k in range(file_count):
		# print("Running gmm by leaving out index {}".format(k))
		(ari_k, gmm_k) = run_gmm_for_g_and_k(file_data, group_count, k,
			config[SKIP_COLS_KEY])
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

def run_gmm_feature_influence(file_data, group_count, skip_lowo, config):
	print("Running gmm for group {}, skipping lowo index: {}".format(
		group_count, skip_lowo))
	if len(file_data) == 0:
		return
	global_skip_cols = config[SKIP_COLS_KEY]
	cols = set(file_data[0].columns) - set(global_skip_cols)
	result = {}
	for c in cols:
		print("Skipping feature {}".format(c))
		skip_cols = global_skip_cols + [c]
		ari_c, gmm_c = run_gmm_for_g_and_k(file_data, group_count, skip_lowo,
			skip_cols = skip_cols)
		result[c] = {
			"ari": ari_c,
			"gmm": gmm_c
		}
	return result

def save_results(output_folder, gmms, selected_g, influence_aris):
	groups = sorted(gmms.keys())
	gmm_result = {}
	for g in groups:
		gmm_result[g] = {k: gmms[g][k] for k in gmms[g].keys() - {"gmm"}}
	selected_result = gmm_result[selected_g]
	influence_result = {
		"group_count": selected_g,
		"lowo_index": selected_result["lowo_index"],
		"ari_with_all_features": selected_result["max_ari"]
	}
	feature_result = {}
	influences = {}
	ari_with_all = selected_result["max_ari"]
	for feature in influence_aris:
		ari = influence_aris[feature]["ari"]
		influences[feature] = {
			"influence": ari_with_all - ari,
			"ari": ari
		}
	feature_result = dict(sorted(influences.items(),
		key=lambda item: item[1]["influence"], reverse=True))
	influence_result["feature_data"] = feature_result
	output = {
		"group_data": gmm_result,
		"selected_group": selected_result,
		"feature_influence": influence_result
	}

	output_path = os.path.join(output_folder, "results.json")
	json_data = json.dumps(output, indent=2)
	with open(output_path, "w") as output_file:
		output_file.write(json_data)
	print("Result saved to {}".format(output_path))

	selected_gmm = gmms[selected_g]["gmm"]
	gmm_path = os.path.join(output_folder, "gmm.joblib")
	joblib.dump(selected_gmm, gmm_path)
	print("GMM model saved to {}".format(gmm_path))

def run_gmm(data_folder, output_folder, config):
	stats_files = fnmatch.filter(os.listdir(data_folder), "{}*.csv".format(
		STATS_PREFIX))
	file_data = []
	for sf in stats_files:
		print("Working on file {} ...".format(sf))
		input_file = os.path.join(data_folder, sf)
		stats_data = pd.read_csv(input_file)
		file_data.append(stats_data)

	gmm_groups = {}
	for g in range(config["group_min"], config["group_max"] + 1):
		result = run_gmm_for_group_count(file_data, g, config)
		gmm_groups[g] = result

	selected_group = max(gmm_groups, key= lambda x: gmm_groups[x]["total_ari"])
	gmm_influence_result = run_gmm_feature_influence(file_data, selected_group,
		gmm_groups[selected_group]["lowo_index"], config)

	save_results(output_folder, gmm_groups, selected_group,
		gmm_influence_result)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the folder containing data files",
		required=True)
	parser.add_argument(
		"--config_path", type=str, help="specifies the json config file",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output folder path",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	config_path = os.path.abspath(args["config_path"])
	output_path = os.path.abspath(args["output_path"])
	with open(config_path) as f:
		config = json.load(f)
	print("Config: {}".format(config))

	run_gmm(data_path, output_path, config)

main()
