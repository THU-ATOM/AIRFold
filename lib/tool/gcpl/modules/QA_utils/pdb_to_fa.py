import argparse
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input_file_pdb", type=str, help="input pdb")
args = parser.parse_args()

file = args.input_file_pdb

flag = True
pdb_path = None

def aa3toaa1(threecode):
    aamap = {'ALA':'A','CYS':'C','CYD':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','MSE':'M','CSO':'C','SEP':'S'}
    return aamap[threecode]

def pdb2fa(pdb,outfa='',gap=True):
    if outfa == '':
        # outfa = pdb[0:-4]+".fa"
        outfa=pdb.replace(pdb.split("/")[-1],"all")+".fa"
    cont = open(pdb)
    savecont = ['>%s\n'%pdb.split("/")[-1][:-4]]
    char = ''
    prvres = -999
    for line in cont:
        if line[:4]!='ATOM':
            continue
        resno = int(line[22:26] )
        if resno == prvres:
            continue
        if resno-prvres > 1 and prvres != -999 and gap:
            # char += '-'*(resno-prvres-1)
            print(" There is a broken link-")
        seq = line[16:20].strip()
        char += aa3toaa1(seq)
        prvres = resno
    char+= '\n'
    savecont.append(char)
    savefile = open(outfa,'a')
    savefile.writelines(savecont)


if file.endswith(".fasta"):
    print("fasta file is already exist,ok")
    flag = False
    exit()

if flag:
    pdb2fa(file)

