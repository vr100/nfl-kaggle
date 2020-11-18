import pandas as pd
import numpy as np
import os, argparse, fnmatch, re, json

TRACK_PREFIX = "week"
GAME_PREFIX = "games"
PLAYER_PREFIX = "players"
PLAY_PREFIX = "plays"

GAME_COL = "gameId"
PLAY_COL = "playId"
PLAYER_COL = "nflId"

PREFIX_DELIM = "_"

NAN_VALUE = "NAN"

NUMERIC_COL = "number"

## appends new_list to old_list without 
## creating duplicates while maintaining existing
## order of old_list
def appendTo(norm_map, value, new_list):
	new_list = sorted(new_list)
	if value not in norm_map:
		norm_map[value] = { NAN_VALUE: 0 }
	old_list = list(norm_map[value].keys())
	only_new_items = [item for item in new_list if item not in old_list]
	index = len(old_list) 
	for item in new_list:
		if item in old_list:
			continue
		norm_map[value][item] = index
		index = index + 1

def normalize_same_type(data, data_format, norm_map):
	if data_format == NUMERIC_COL:
		data = data.replace([None, "NAN"], "0")
		data = data.astype(np.int8)
		return data

	data = data.fillna(NAN_VALUE)
	columns = list(data.columns)
	for col in columns:
		unique_values = list(data[col].unique())
		appendTo(norm_map, data_format, unique_values)
		data = data.replace({col: norm_map[data_format]})
	data = data.astype(np.int8)
	return data

def split_data(data, col, split, norm_map):
	new_data = (data[col]
		.str.split(split["delim"])
		.str.join(" ")
		.str.strip()
		.str.replace(r"\s+", " ")
		.str.split(" ", expand=True))
	col_count = len(list(new_data.columns))
	format_len = len(split["format"])
	for start in range(format_len):
		prefix = (split["prefix"] + PREFIX_DELIM 
			+ split["format"][start] + PREFIX_DELIM)
		selected_columns = list(range(start, col_count, format_len))
		cols_map = {}
		for index in range(len(selected_columns)):
			cols_index = start + index * format_len
			cols_map[cols_index] = prefix + str(index)
		new_data.rename(columns = cols_map, inplace=True)
		new_cols = list(cols_map.values())
		new_data[new_cols] = normalize_same_type(new_data[new_cols],
			split["format"][start], norm_map)
	data = data.drop(columns=[col])
	data[list(new_data.columns)] = new_data
	return data

def normalize_data(data, meta, norm_map, split={}):
	object_cols = list(data.select_dtypes(["object"]).columns)
	data[object_cols] = data[object_cols].fillna(NAN_VALUE)
	boolean_cols = list(data.select_dtypes(["bool"]).columns)
	data[boolean_cols] = data[boolean_cols].astype(np.int8)

	for col in meta:
		unique_values = list(data[col].unique())
		value = meta[col]
		appendTo(norm_map, value, unique_values)
		data = data.replace({col: norm_map[value]})

	meta_cols = list(meta.keys())
	data[meta_cols] = data[meta_cols].astype(np.int8)

	for col in split:
		data = split_data(data, col, split[col], norm_map)

	data.reset_index(drop=True, inplace=True)
	return data

def normalize_play(data, normalize_map):
	SKIP_COLS = ["playDescription", "gameClock"]
	data = data.drop(columns=SKIP_COLS)
	normalize_split = {
		"personnelO": {
			"delim": ",",
			"prefix": "personnel_o",
			"format": ["number", "positions"],
			"style": "fixed"
		},
		"personnelD": {
			"delim": ",",
			"prefix": "personnel_d",
			"format": ["number", "positions"],
			"style": "fixed"
		},
		"penaltyCodes": {
			"delim": ";",
			"prefix": "penalty_code",
			"format": ["codes"]
		},
		"penaltyJerseyNumbers": {
			"delim": ";",
			"prefix": "penalty_jn",
			"format": ["teams", "number"]
		}
	}
	normalize_meta = {
		"possessionTeam": "teams",
		"playType": "playtypes",
		"yardlineSide": "teams",
		"offenseFormation": "formations",
		"typeDropback": "dropbacktypes",
		"passResult": "results"
	}
	return normalize_data(data, normalize_meta, normalize_map,
		split=normalize_split)

def normalize_game(data, normalize_map):
	SKIP_COLS = ["gameDate", "gameTimeEastern", "week"]
	data = data.drop(columns=SKIP_COLS)
	normalize_meta = {
		"homeTeamAbbr": "teams",
		"visitorTeamAbbr": "teams"
	}
	return normalize_data(data, normalize_meta, normalize_map)

def normalize_track(data, normalize_map):
	SKIP_COLS = ["time", "displayName"]
	data = data.drop(columns=SKIP_COLS)
	normalize_meta = {
		"event": "events",
		"position": "positions",
		"team": "teamtypes",
		"playDirection": "directions",
		"route": "routes"
	}
	return normalize_data(data, normalize_meta, normalize_map)

def save_dataframe_as_json(dataframe, output_path, filename):
	json_data = dataframe.to_json(orient="records", indent=2)
	json_path = os.path.join(output_path, filename)
	with open(json_path, "w") as output:
		output.write(json_data)

def get_dataframes(data_path, output_path):
	normalizer = {}

	game_path = os.path.join(data_path, "{}.csv".format(GAME_PREFIX))
	game_data = pd.read_csv(game_path)
	save_dataframe_as_json(game_data, output_path,
		"{}.json".format(GAME_PREFIX))
	game_data = normalize_game(game_data, normalizer)

	player_path = os.path.join(data_path, "{}.csv".format(PLAYER_PREFIX))
	player_data = pd.read_csv(player_path)
	save_dataframe_as_json(player_data, output_path,
		"{}.json".format(PLAYER_PREFIX))

	play_path = os.path.join(data_path, "{}.csv".format(PLAY_PREFIX))
	play_data = pd.read_csv(play_path)
	play_data = normalize_play(play_data, normalizer)

	track_files = fnmatch.filter(os.listdir(data_path), "{}*.csv".format(
		TRACK_PREFIX))

	index = 0
	for tf in track_files:
		track_path = os.path.join(data_path, tf)
		track_data = pd.read_csv(track_path)
		track_data = normalize_track(track_data, normalizer)

		join_data = pd.merge(track_data, play_data, on=[GAME_COL, PLAY_COL], how="left")
		join_data = pd.merge(join_data, game_data, on=[GAME_COL], how="left")

		join_output_path = os.path.join(output_path, "{}.pkl".format(index))
		join_data.to_pickle(join_output_path)
		index = index + 1
		print("Save join data for {} to {}".format(tf, join_output_path))

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

	get_dataframes(data_path, output_path)

main()
