# Metrics defined in paper - https://arxiv.org/abs/1906.11373
# "Unsupervised Methods for Identifying Pass Coverage Among Defensive Backs with NFL Player Tracking Data"

import argparse, os, fnmatch, math
import pandas as pd

TRACK_PREFIX = "week"
GAME_FILE = "games"
PLAY_FILE = "plays"

GAME_ID = "gameId"
PLAY_ID = "playId"
FRAME_ID = "frameId"
POSITION_FLD = "position"
TEAM_FLD = "team"
OFFENSE_FLD = "possessionTeam"
HOME_FLD = "homeTeamAbbr"
AWAY_FLD = "visitorTeamAbbr"
NFL_ID = "nflId"
X = "x"
Y = "y"
SPEED = "s"
DIR = "dir"
EVENT = "event"

HOME_TEAM = "home"
AWAY_TEAM = "away"
FOOTBALL = "football"
CB_VAL = "CB"
SNAP_EVENT = "ball_snap"
PASS_EVENTS = ["pass_forward", "pass_shovel"]

S_OFFENSE = "offense"
S_DEFENSE = "defense"
S_X = "x"
S_Y = "y"
S_SPEED = "speed"
S_DIR = "dir"
S_DIST_OFF = "dist_off"
S_DIST_DEF = "dist_def"
S_DIST_OFF_DEF = "dist_between_off_def"
S_DIR_OFF = "dir_off"
S_FB_CLOSEST = "closest_to_football"
S_EVENT = "event"

A_X = "x"
A_Y = "y"
A_S = "speed"
A_DO = "dist_off"
A_DD = "dist_def"
A_DIRO = "dir_off"
A_R = "ratio"
A_CLOSEST = "closest_frames"

A_MEAN_PREFIX = "mean_"
A_VAR_PREFIX = "var_"

A_BS_PREFIX = "before_snap_"
A_AS_PREFIX = "after_snap_"
A_BP_PREFIX = "before_pass_"
A_AP_PREFIX = "after_pass_"
A_BSP_PREFIX = "between_snap_pass_"
A_FULL_PREFIX = "full_"

A_S_PREFIX = "at_snap_"
A_P_PREFIX = "at_pass_"
A_M_PREFIX = "at_mid_"

NO_VALUE = -1000

def compute_common_stats(data, game, play):
	stats = {}
	data = data[(data[GAME_ID] == game) & (data[PLAY_ID] == play)]
	off_team = data[OFFENSE_FLD].values[0]
	home_team = data[HOME_FLD].values[0]
	away_team = data[AWAY_FLD].values[0]
	stats[S_OFFENSE] = HOME_TEAM if off_team == home_team else AWAY_TEAM
	stats[S_DEFENSE] = AWAY_TEAM if off_team == home_team else HOME_TEAM
	return stats

def get_nearest(player, players):
	x = player[X]
	y = player[Y]
	nearest = None
	min_dist = float("inf")
	for _,p in players.iterrows():
		px = p[X]
		py = p[Y]
		square_dist = float((x - px) ** 2 + (y - py) ** 2)
		if square_dist < min_dist:
			min_dist = square_dist
			nearest = p
	return nearest

def get_dist(player1, player2):
	if player1 is None or player2 is None: return NO_VALUE
	x1 = player1[X]
	y1 = player1[Y]
	x2 = player2[X]
	y2 = player2[Y]
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def get_dir_diff(player1, player2):
	if player1 is None or player2 is None: return NO_VALUE
	return (player1[DIR] - player2[DIR])

def get_empty_stats():
	stats = {
		S_X : [],
		S_Y : [],
		S_SPEED : [],
		S_DIR: [],
		S_DIST_OFF: [],
		S_DIST_DEF: [],
		S_DIST_OFF_DEF: [],
		S_DIR_OFF: [],
		S_FB_CLOSEST: [],
		S_EVENT: []
	}
	return stats

def append_stats(stats, new_stats):
	for s in new_stats:
		stats[s].append(new_stats[s])

def update_stats(stats, player, oPlayer, dPlayer, closest_to_football):
	new_stats = {
		S_X : player[X],
		S_Y : player[Y],
		S_SPEED : player[SPEED],
		S_DIR: player[DIR],
		S_DIST_OFF: get_dist(player, oPlayer),
		S_DIST_DEF: get_dist(player, dPlayer),
		S_DIST_OFF_DEF: get_dist(oPlayer, dPlayer),
		S_DIR_OFF: get_dir_diff(oPlayer, player),
		S_FB_CLOSEST: closest_to_football,
		S_EVENT: player[EVENT]
	}
	append_stats(stats, new_stats)

def get_stats_for_frame(data, common_stats, stats, frame):
	offense = data[data[TEAM_FLD] == common_stats[S_OFFENSE]]
	defense = data[data[TEAM_FLD] == common_stats[S_DEFENSE]]
	cornerback = defense[defense[POSITION_FLD] == CB_VAL]
	football = data[data[TEAM_FLD] == FOOTBALL].head(1)
	nearest_ball = get_nearest(football, cornerback) if len(football) == 1 \
		else None
	for _,cb in cornerback.iterrows():
		defense_without_cb = defense[defense[NFL_ID] != cb[NFL_ID]]
		nearest_off = get_nearest(cb, offense)
		nearest_def = get_nearest(cb, defense_without_cb)
		closest_to_football = 1 if nearest_ball is not None and \
			(cb[NFL_ID] == nearest_ball[NFL_ID]) else 0
		cbId = cb[NFL_ID]
		cb_stats = stats[cbId] if cbId in stats else get_empty_stats()
		update_stats(cb_stats, cb, nearest_off, nearest_def,
			closest_to_football)
		stats[cbId] = cb_stats
	return stats

def is_valid(x):
	return (x!= NO_VALUE and not math.isnan(x) and x is not None)

def split_by_events(data, events):
	snap_index = events.index(SNAP_EVENT) if SNAP_EVENT in events else - 1
	for evt in PASS_EVENTS:
		pass_index = events.index(evt) if evt in events else -1
		if pass_index != -1:
			break
	if snap_index == -1 or pass_index == -1 or pass_index < snap_index:
		return []
	before_snap = slice(0, snap_index)
	between_snap_pass = slice(snap_index, pass_index + 1)
	after_pass = slice(pass_index + 1, len(data))
	new_data = [data[before_snap], data[between_snap_pass], data[after_pass]]
	return new_data

def set_mean_variance(stats, data, events, dataname):
	split_data = split_by_events(data, events)
	# before snap, after snap, between snap and pass,
	# before pass, after pass, full
	data_map = {
		A_BS_PREFIX : split_data[0],
		A_AS_PREFIX: split_data[1] + split_data[2],
		A_BSP_PREFIX: split_data[1],
		A_BP_PREFIX: split_data[0] + split_data[1],
		A_AP_PREFIX: split_data[2],
		A_FULL_PREFIX: split_data[0] + split_data[1] + split_data[2]
	}

	for item in data_map:
		mean,variance = get_mean_variance(data_map[item])
		item_mean_key = item + A_MEAN_PREFIX + dataname
		item_var_key = item + A_VAR_PREFIX + dataname
		stats[item_mean_key] = mean
		stats[item_var_key] = variance

def get_mean_variance(data):
	data = [x for x in data if is_valid(x)]
	if len(data) == 0:
		return 0, 0
	mean = sum(data) / len(data)
	variance = sum((i - mean) ** 2 for i in data) / len(data)
	return mean, variance

def set_ratio_mean_variance(stats, num, den, events, dataname):
	full_ratio = get_ratio(num, den)
	mean, variance = get_mean_variance(full_ratio)
	num_split = split_by_events(num, events)
	den_split = split_by_events(den, events)
	snap_value =  get_ratio_value(num_split[1][0], den_split[1][0])
	pass_value = get_ratio_value(num_split[1][-1], den_split[1][-1])
	mid_index = math.ceil(len(num_split[1]) / 2) - 1
	mid_value = get_ratio_value(num_split[1][mid_index], den_split[1][mid_index])
	stats[A_FULL_PREFIX + A_MEAN_PREFIX + dataname] = mean
	stats[A_FULL_PREFIX + A_VAR_PREFIX + dataname] = variance
	stats[A_S_PREFIX + dataname] = snap_value
	stats[A_P_PREFIX + dataname] = pass_value
	stats[A_M_PREFIX + dataname] = mid_value

def get_ratio_value(n, d):
	return n/d if (is_valid(n) and is_valid(d) and d != 0) else 0

def get_ratio(num, den):
	zipped = [(n,d) for (n,d) in zip(num, den) if (is_valid(n) and \
		is_valid(d) and d != 0 ) ]
	return [n/d for (n,d) in zipped]

def gather_frame_stats(frame_stats, game, play):
	data = pd.DataFrame()
	for player in frame_stats:
		stats = frame_stats[player]
		if (not any(x in stats[S_EVENT] for x in PASS_EVENTS)):
			continue
		events = stats[S_EVENT]
		new_stats = {
			NFL_ID: player,
			GAME_ID: game,
			PLAY_ID: play,
			A_CLOSEST: 	sum(stats[S_FB_CLOSEST])
		}
		set_mean_variance(new_stats, stats[S_X], events, A_X)
		set_mean_variance(new_stats, stats[S_Y], events, A_Y)
		set_mean_variance(new_stats, stats[S_SPEED], events, A_S)
		set_mean_variance(new_stats, stats[S_DIST_OFF], events, A_DO)
		set_mean_variance(new_stats, stats[S_DIST_DEF], events, A_DD)
		set_mean_variance(new_stats, stats[S_DIR_OFF], events, A_DIRO)
		set_ratio_mean_variance(new_stats, stats[S_DIST_OFF],
			stats[S_DIST_OFF_DEF], events, A_R)
		data = data.append(new_stats, ignore_index=True)
	return data

def compute_stats_for_play(data, game, play, common_data):
	frames = sorted(data[FRAME_ID].unique())
	# print("Total frames: {}".format(len(frames)))
	common_stats = compute_common_stats(common_data, game, play)
	stats = {}
	for frame in frames:
		frame_data = data[data[FRAME_ID] == frame]
		frame_data = frame_data.sort_values(by=FRAME_ID)
		get_stats_for_frame(frame_data, common_stats, stats, frame)
	stats_data = gather_frame_stats(stats, game, play)
	return stats_data

def compute_stats_for_game(data, game, common_data):
	plays = sorted(data[PLAY_ID].unique())
	stats_data = pd.DataFrame()
	for play in plays:
		# print("Processing play {} ...".format(play))
		play_data = data[data[PLAY_ID] == play]
		play_stats = compute_stats_for_play(play_data, game, play,
			common_data)
		stats_data = stats_data.append(play_stats, ignore_index=True)
	return stats_data

def compute_stats_for_file(filename, data_folder, output_folder, common_data):
	file_path = os.path.join(data_folder, filename)
	output_file = os.path.join(output_folder, filename)
	stats = pd.DataFrame()
	data = pd.read_csv(file_path)
	games = sorted(data[GAME_ID].unique())
	for game in games:
		print("Processing game {} ...".format(game))
		game_data = data[data[GAME_ID] == game]
		game_stats = compute_stats_for_game(game_data, game,
			common_data)
		stats = stats.append(game_stats, ignore_index=True)
	stats.to_csv(output_file)

def compute_stats(data_folder, output_folder):
	game_file = os.path.join(data_folder, "{}.csv".format(GAME_FILE))
	play_file = os.path.join(data_folder, "{}.csv".format(PLAY_FILE))
	game_data = pd.read_csv(game_file)
	play_data = pd.read_csv(play_file)
	common_data = pd.merge(play_data, game_data, on=[GAME_ID], how="left")

	track_files = fnmatch.filter(os.listdir(data_folder), "{}*.csv".format(
		TRACK_PREFIX))
	for tf in track_files:
		print("Working on file {} ...".format(tf))
		compute_stats_for_file(tf, data_folder, output_folder, common_data)

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

	compute_stats(data_path, output_path)

main()