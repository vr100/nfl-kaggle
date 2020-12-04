import pandas as pd
import matplotlib.pyplot as plot
import os, argparse

FIG_SIZE=(8,12)
Y_LIM=(0, 500)


def plot_graph(input_path, output_path):
	data = pd.read_csv(input_path)
	axes = data.plot.bar(stacked=True, figsize=FIG_SIZE)
	axes.set_ylim(Y_LIM[0], Y_LIM[1])
	plot.savefig(output_path)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_path", type=str, help="specifies the input csv file path",
		required=True)
	parser.add_argument(
		"--output_path", type=str, help="specifies the output image file path",
		required=True)
	return vars(parser.parse_args())

def main():
	args = parse_args()
	print("Args: {}".format(args))
	data_path = os.path.abspath(args["data_path"])
	output_path = os.path.abspath(args["output_path"])

	plot_graph(data_path, output_path)
	print("Image file saved to {}".format(output_path))

main()