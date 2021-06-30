import argparse 
from src.create_results import *

'''
Analyze results that are saved as .pkl files in results/raw.
'''
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="Name of the dataset.")
ap.add_argument("-d", "--directory", required=True, help="Directory of pickle files.")
ap.add_argument("-l", "--lassos", required=True, help="Number of lassos.")
ap.add_argument("-s", "--step", required=False, default=1, help="Step between lassos.")
args = vars(ap.parse_args())

dataset = args["name"] # dataset name 
directory = args["directory"] # directory name
lassos = args["lassos"] # number of lassos
step = args["step"] # step number

def main(dataset, directory, lassos, step):
    create_results(dataset, directory, lassos, step)
      
if __name__ == "__main__":
    main(dataset, directory, lassos, step)