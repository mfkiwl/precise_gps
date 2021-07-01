import argparse 
from src.create_results import *

'''
Analyze results that are saved as .pkl files in results/raw.

Args:
    name (string)      : dataset name
    directory (string) : directory name 
    lassos (int)       : number of lassos for some plots (MLL)
    step (int)         : steps between lassos (see above)
'''
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="Name of the dataset.")
ap.add_argument("-d", "--directory", nargs = '+', required=True, help="Directory of pickle files.")
ap.add_argument("-l", "--lassos", type=int, required=True, help="Number of lassos.")
ap.add_argument("-s", "--step", type=int, required=False, default=1, help="Step between lassos.")
ap.add_argument("-v", "--visualize", type=bool, required=False, default=0, help="Wheter plots are shown during running the program.")
args = vars(ap.parse_args())

dataset = args["name"] # dataset name 
directory = args["directory"] # directory name
lassos = args["lassos"] # number of lassos
step = args["step"] # step number
show = args["visualize"] # whether plots are shown during running the program

def main(dataset, directory, lassos, step):
    create_results(dataset, directory, lassos, step, show)
      
if __name__ == "__main__":
    main(dataset, directory, lassos, step)