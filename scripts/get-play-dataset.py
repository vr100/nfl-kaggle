import pandas as pd
import numpy as np
import os, argparse, json

GAME_PREFIX = "games"
PLAY_PREFIX = "plays"
NORMALIZE_PREFIX = "normalizer"
X_PREFIX = "x"
Y_PREFIX = "y"

NAN_VALUE = "NAN_VALUE"
OFFENSE_VALUE = "OFF"
DEFENSE_VALUE = "DEF"
BOTH_VALUE = "BOTH"

GAME_COL = "gameId"
JERSEY_COL = "penaltyJerseyNumbers"
PENALTY_COL = "penaltyOwner"
OFFENSE_TEAM_COL = "possessionTeam"

def get_indexed_dictionary(new_list):
	index_dict = { NAN_VALUE: 0 }
	index = 1
	for item in new_list:
		if item in index_dict:
			continue
		index_dict[item] = index
		index = index + 1
	return index_dict

def normalize_data(data, meta, norm_map):
	object_cols = list(data.select_dtypes(["object"]).columns)
	data[object_cols] = data[object_cols].fillna(NAN_VALUE)
	boolean_cols = list(data.select_dtypes(["bool"]).columns)
	data[boolean_cols] = data[boolean_cols].astype(np.int8)

	for col in meta:
		unique_values = list(data[col].unique())
		value = meta[col]
		indexed_dict = get_indexed_dictionary(unique_values)
		data = data.replace({col: indexed_dict})
		norm_map[value] = indexed_dict

	meta_cols = list(meta.keys())
	data[meta_cols] = data[meta_cols].astype(np.int8)

	data.reset_index(drop=True, inplace=True)
	return data

def add_penalty_owner(data):
	data[JERSEY_COL] = (data[JERSEY_COL]
		.str.split(";")
		.str.join(" ")
		.str.replace(r"\d+", "")
		.str.replace(r"\s+", " ")
		.str.strip()
		.str.split(" "))

	data[JERSEY_COL] = data[JERSEY_COL].dropna().apply(set)

	data[PENALTY_COL] = NAN_VALUE
	data[PENALTY_COL] = data.apply(lambda x: NAN_VALUE
		if pd.isnull(x[JERSEY_COL]) else
		(BOTH_VALUE if len(x[JERSEY_COL]) > 1 else
		(OFFENSE_VALUE if list(x[JERSEY_COL])[0] == x[OFFENSE_TEAM_COL]
			else DEFENSE_VALUE)), axis = 1)
	return data

def get_dataset(data_path, output_path):
	normalizer = {}

	play_path = os.path.join(data_path, "{}.csv".format(PLAY_PREFIX))
	play_data = pd.read_csv(play_path)

	play_data = add_penalty_owner(play_data)

	play_data = play_data[["playType", "offenseFormation",
		"defendersInTheBox", "numberOfPassRushers", "typeDropback",
		"passResult", "isDefensivePI", PENALTY_COL]]

	normalize_meta = {
		"playType": "playtypes",
		"offenseFormation": "formations",
		"typeDropback": "dropbacktypes",
		"passResult": "results",
		PENALTY_COL: "teamtypes",
	}

	play_data = normalize_data(play_data, normalize_meta, normalizer)

	x_play_data = play_data[["playType", "offenseFormation",
		"defendersInTheBox", "numberOfPassRushers", "typeDropback"]]
	x_file = os.path.join(output_path, "{}.csv".format(X_PREFIX))
	x_play_data.to_csv(x_file, index=False)

	y_result = play_data[["passResult"]]
	y_file = os.path.join(output_path, "{}-{}.csv".format(
		Y_PREFIX, "passResult"))
	y_result.to_csv(y_file, index=False)

	y_defensive_pi = play_data[["isDefensivePI"]]
	y_file = os.path.join(output_path, "{}-{}.csv".format(
		Y_PREFIX, "isDefensivePI"))
	y_defensive_pi.to_csv(y_file, index=False)

	y_penalty_owner = play_data[[PENALTY_COL]]
	y_file = os.path.join(output_path, "{}-{}.csv".format(
		Y_PREFIX, PENALTY_COL))
	y_penalty_owner.to_csv(y_file, index=False)

	json_data = json.dumps(normalizer, indent=2)
	normalizer_path = os.path.join(output_path, "{}.json".format(
		NORMALIZE_PREFIX))
	with open(normalizer_path, "w") as output:
		output.write(json_data)

	print("Saved data files and normalize map to {}".format(output_path))

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

	get_dataset(data_path, output_path)

main()
