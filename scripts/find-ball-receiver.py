import argparse, os, fnmatch, math
import pandas as pd
from scipy import stats as scipystats

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
X = "x"
Y = "y"
EVENT = "event"
NFL_ID = "nflId"

SNAP_EVENT = "ball_snap"
PASS_EVENTS = ["pass_forward", "pass_shovel"]
BALL = "football"
HOME_TEAM = "home"
AWAY_TEAM = "away"

S_OFFENSE = "offense"
S_DEFENSE = "defense"
PLAYER = "receiver"
DIFF = "diff"
DEFENDENT_PREFIX = "def_"
DEFENDENT_DIST_PREFIX = "def_dist_"
RANK = "rank"

FRAME_COUNT = 5
MAX_RECEIVERS = 3
YARDS_AROUND = 10
MAX_DEFENDENTS = 2

def get_basename(filename):
	return os.path.splitext(os.path.basename(filename))[0]

def find_offense_defense(data, game, play):
	stats = {}
	data = data[(data[GAME_ID] == game) & (data[PLAY_ID] == play)]
	off_team = data[OFFENSE_FLD].values[0]
	home_team = data[HOME_FLD].values[0]
	away_team = data[AWAY_FLD].values[0]
	stats[S_OFFENSE] = HOME_TEAM if off_team == home_team else AWAY_TEAM
	stats[S_DEFENSE] = AWAY_TEAM if off_team == home_team else HOME_TEAM
	return stats

def compute_distance(line, point):
	# line: ax + by + c = 0
	# point: (x1, y1)
	# distance = abs(a * x1 + b * y1 + c) / sqrt(a ^ 2 + b ^ 2)
	num = abs(line["a"] * point["x"] + line["b"] * point["y"] + line["c"])
	den = math.sqrt(line["a"] ** 2 + line["b"] ** 2)
	return (num / den)

def compute_distance_from_point(point1, point2):
	if point1 is None or point2 is None: return NO_VALUE
	x1 = point1["x"]
	y1 = point1["y"]
	x2 = point2["x"]
	y2 = point2["y"]
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def distance_diff_from_ball(data, start, end, stats, line):
	start_data = data[(data[FRAME_ID] == start) &
		(data[TEAM_FLD] == stats[S_OFFENSE])]
	end_data = data[(data[FRAME_ID] == end) &
		(data[TEAM_FLD] == stats[S_OFFENSE])]
	start_ball = data[(data[FRAME_ID] == start) & (data[TEAM_FLD] == BALL)]
	end_ball = data[(data[FRAME_ID] == end) & (data[TEAM_FLD] == BALL)]
	if len(start_ball) == 0 or len(end_ball) == 0:
		return None
	start_position = {
		"x": start_ball[X].values[0],
		"y": start_ball[Y].values[0]
	}
	start_distance = {}
	for _, player in start_data.iterrows():
		field_position = { "x": player[X], "y": player[Y] }
		line_distance = compute_distance(line, field_position)
		if line_distance > YARDS_AROUND:
			continue
		distance = compute_distance_from_point(start_position, field_position)
		player_id = player[NFL_ID]
		start_distance[player_id] = distance
	end_position = {
		"x": end_ball[X].values[0],
		"y": end_ball[Y].values[0]
	}
	end_distance = {}
	for _, player in end_data.iterrows():
		field_position = { "x": player[X], "y": player[Y] }
		line_distance = compute_distance(line, field_position)
		if line_distance > YARDS_AROUND:
			continue
		distance = compute_distance_from_point(end_position, field_position)
		player_id = player[NFL_ID]
		end_distance[player_id] = distance
	distance_between_balls = compute_distance_from_point(start_position,
		end_position)
	diff_distance = {}
	for player in start_distance:
		if player not in end_distance:
			continue
		diff = start_distance[player] - end_distance[player]
		if diff < 0:
			continue
		diff_distance[player] = abs(abs(diff) - distance_between_balls)

	diff_distance = dict(sorted(diff_distance.items(),key=lambda item: item[1]))
	return diff_distance

def get_nearest(player, players, max_k):
	player_dict = {}
	total_frames = 0
	for _,frame in player.iterrows():
		total_frames += 1
		frame_id = frame[FRAME_ID]
		x = frame[X]
		y = frame[Y]
		frame_players= players[players[FRAME_ID] == frame_id]
		for _,p in frame_players.iterrows():
			px = p[X]
			py = p[Y]
			square_dist = math.sqrt(float((x - px) ** 2 + (y - py) ** 2))
			player_id = p[NFL_ID]
			if player_id in player_dict:
				player_data = player_dict[player_id]
			else:
				player_data = {
					"total": 0,
					"count": 0
				}
			player_data["total"] += square_dist
			player_data["count"] += 1
			player_dict[player_id] = player_data

	avg_distance = {}
	max_frames_expected = total_frames / 2
	for player in player_dict:
		player_data = player_dict[player]
		if player_data["count"]< max_frames_expected:
			continue
		avg_distance[player] = player_data["total"] / player_data["count"]

	avg_distance = dict(sorted(avg_distance.items(),key=lambda item: item[1]))
	nearest_k_players = {k: avg_distance[k] for k in list(avg_distance)[:max_k]}
	return nearest_k_players

def get_closest_defendents(data, players, frame, stats):
	defense = data[data[TEAM_FLD] == stats[S_DEFENSE]]
	offense = data[data[TEAM_FLD] == stats[S_OFFENSE]]
	frame_offense = offense[offense[FRAME_ID] == frame]
	closest_defendents = {}
	for _,player in frame_offense.iterrows():
		player_id = player[NFL_ID]
		if player_id not in players:
			continue
		player_offense = offense[offense[NFL_ID] == player_id]
		closest_defendents[player_id] = get_nearest(player_offense,
			defense, MAX_DEFENDENTS)
	return closest_defendents

def compute_for_play(data, game, play, common_data):
	data = data.sort_values(by=FRAME_ID)
	stats = find_offense_defense(common_data, game, play)
	pass_frame = -1
	for event in PASS_EVENTS:
		temp_data = data[data[EVENT] == event]
		if len(temp_data) != 0:
			pass_frame = temp_data[FRAME_ID].unique()[0]
			break
	if pass_frame == -1:
		return None
	frames = range(pass_frame, pass_frame + FRAME_COUNT)
	ball_x = []
	ball_y = []
	last_valid_frame = -1
	for f in frames:
		ball_data = data[(data[FRAME_ID] == f) & (data[TEAM_FLD] == BALL)]
		if len(ball_data) == 0:
			continue
		ball_x.append(ball_data[X].head(1))
		ball_y.append(ball_data[Y].head(1))
		last_valid_frame = f
	if len(ball_x) < 2 or len(ball_y) < 2:
		return None
	result = scipystats.mstats.linregress(ball_x, ball_y)
	slope, intercept = result[:2]
	# y = mx + c can be rewritten in the form ax + by + c = 0
	# as  mx - y + c = 0
	ball_line = { "a": slope, "b": -1, "c": intercept}
	ball_distance = distance_diff_from_ball(data, pass_frame, last_valid_frame,
		stats, ball_line)

	top_closest_players = {k: ball_distance[k] \
		for k in list(ball_distance)[:MAX_RECEIVERS]}
	closest_defendents = get_closest_defendents(data, top_closest_players,
		pass_frame, stats)
	data_dict = {
		GAME_ID: game,
		PLAY_ID: play,
		"line_a": ball_line["a"],
		"line_b": ball_line["b"],
		"line_c": ball_line["c"]
	}
	row_list = []
	count = 0
	for player in top_closest_players:
		player_dict = {
			RANK: count,
			PLAYER: player,
			DIFF: top_closest_players[player]
		};
		defendents = closest_defendents[player]
		def_count = 0
		for defendent in defendents:
			key = "{}{}".format(DEFENDENT_PREFIX, def_count)
			player_dict[key] = defendent
			key = "{}{}".format(DEFENDENT_DIST_PREFIX, def_count)
			player_dict[key] = defendents[defendent]
			def_count += 1
		count += 1
		row_list.append({**data_dict, **player_dict})
	if count == 0:
		return None
	return pd.DataFrame(row_list)

def compute_for_game(data, game, common_data):
	plays = sorted(data[PLAY_ID].unique())
	receiver_data = pd.DataFrame()
	for play in plays:
		# print("Processing play {} ...".format(play))
		play_data = data[data[PLAY_ID] == play]
		pr_data = compute_for_play(play_data, game, play, common_data)
		if pr_data is not None:
			receiver_data = receiver_data.append(pr_data, ignore_index=True)
	return receiver_data

def compute_for_file(filename, data_folder, output_folder, common_data):
	file_path = os.path.join(data_folder, filename)
	receiver_data = pd.DataFrame()
	data = pd.read_csv(file_path)
	games = sorted(data[GAME_ID].unique())
	for game in games:
		print("Processing game {} ...".format(game))
		game_data = data[data[GAME_ID] == game]
		gr_data = compute_for_game(game_data, game, common_data)
		receiver_data = receiver_data.append(gr_data, ignore_index=True)
	output_file = os.path.join(output_folder, "{}.json".format(
		get_basename(filename)))
	receiver_data.to_json(output_file, orient="records", indent=4)

def compute_ball_receiver(data_folder, output_folder):
	game_file = os.path.join(data_folder, "{}.csv".format(GAME_FILE))
	play_file = os.path.join(data_folder, "{}.csv".format(PLAY_FILE))
	game_data = pd.read_csv(game_file)
	play_data = pd.read_csv(play_file)
	common_data = pd.merge(play_data, game_data, on=[GAME_ID], how="left")

	track_files = fnmatch.filter(os.listdir(data_folder), "{}*.csv".format(
		TRACK_PREFIX))
	for tf in track_files:
		print("Working on file {} ...".format(tf))
		compute_for_file(tf, data_folder, output_folder, common_data)

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

	compute_ball_receiver(data_path, output_path)

main()
