# parse pdb
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_dir", type=str, help="input dirs")
args = parser.parse_args()

input_dir= args.input_dir

input_dir_list=os.listdir(input_dir)

script_dir=os.path.abspath(os.path.dirname(__file__))
for pdb in input_dir_list:
    dir_pdb = input_dir + "/" +pdb
    if pdb.endswith(".pdb"):
        cmd = "python %s/pdb_to_fa.py %s "%(script_dir, dir_pdb)
        os.system(cmd)


