import argparse 
from src.create_results import create_results

'''
Analyze results that are saved as .pkl files in results/raw.

Args:
    name (str) : dataset name
    directory (str) : directory name 
    lassos (int) : number of lassos for some plots (MLL)
    step (int) : steps between lassos (see above)
'''
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="Name of the dataset.")
ap.add_argument("-d", "--directory", nargs = '+', required=True, 
                help="Directory of pickle files.")
ap.add_argument("-l", "--lassos", type=int, required=True, 
                help="Number of lassos.")
ap.add_argument("-s", "--step", type=int, required=False, default=1, 
                help="Step between lassos.")
ap.add_argument("-v", "--visualize", type=bool, required=False, default=0, 
                help="Wheter plots are shown during running the program.")
ap.add_argument("-loss", "--loss_landscape", type=bool, required=False, 
                default=0, help="Whether loss landscape plots are formed.")
args = vars(ap.parse_args())

dataset = args["name"]
directory = args["directory"]
lassos = args["lassos"]
step = args["step"]
show = args["visualize"]
loss_landscape_arg = args["loss_landscape"]

def main(dataset, directory, lassos, step):
    create_results(dataset, directory, lassos, step, show, loss_landscape_arg)
      
if __name__ == "__main__":
    main(dataset, directory, lassos, step)