#!/usr/bin/env python3
import os
import sys
import re
import subprocess
from config import para_json
from utils import job_queue, submitjob, command_header, fasta2len

docstring = '''
MSA_selection.py 

    Ranking MSAs by scoring function, the scoring function could be 
    DeepPotential top 10L sum contact probablity or AlphaFold2 pLDDT score. 
    
Usage: MSA_selection [option]

    required options:
    -i=/home/simth/seq.txt
    -o=/home/simth/test

    optional options:
    -run_type=local (default) or sbatch
    -method=deeppotential or alphafold2 (default)

'''


if __name__ == "__main__":

    run_type_arg = ""
    inputsourcefile = ""
    outputdirectory = ""
    method_arg="alphafold2"  ### deeppotential or alphafold2

    argv = []
    for arg in sys.argv[1:]:
        if arg.startswith("-i="):
            inputsourcefile = os.path.abspath(arg[len("-i="):])
        elif arg.startswith("-o="):
            outputdirectory = os.path.abspath(arg[len("-o="):])
        elif arg.startswith("-run_type="):
            run_type_arg = arg[len("-run_type="):]
        elif arg.startswith("-method="):
            method_arg = arg[len("-method="):]
        elif arg.startswith('-'):
            sys.stderr.write("ERROR! No such option %s\n" % arg)
            exit()
        else:
            argv.append(arg)

    if len(argv) != 0:
        sys.stderr.write(docstring)
        exit()

    user = os.environ['USER']
    rootdir = os.path.dirname(os.path.realpath(__file__))
    bindir = f"{rootdir}/bin"

    if outputdirectory != "":
        outdir = outputdirectory
    else:
        print("output directory parameter -o is required!")
        sys.stderr.write(docstring)
        exit(1)

    if inputsourcefile != "":
        if os.path.isfile(f"{inputsourcefile}"):
            if not os.path.exists(os.path.join(outdir,"seq.txt")):
                os.system(f"cp {inputsourcefile} {outdir}/seq.txt")
            if not os.path.exists(os.path.join(outdir,"seq.fasta")):
                os.system(f"cp {inputsourcefile} {outdir}/seq.fasta")
        else:
            print("Input sequence file parameter -i is required!")
            sys.stderr.write(docstring)
            exit(2)

    if run_type_arg != "":
        run_type = run_type_arg
    else:
        run_type = para_json['run_type']

    if run_type == "sbatch":
        run_type = 2
    else:
        run_type = 1  ## local

    #print(run_type)
    partition = para_json['partition']
    account = para_json['account']
    Q = "urgent"
    if Q == "default":
        Q = "normal"

    dcpu = para_json['dMSAcpu']
    python3=para_json['python_DeepPotential']
    s = os.path.basename(outdir)
    if method_arg == "deeppotential":
        print("Start doing MSAs ranking by DeepPotential\n")
        qzy = job_queue(run_type)

        
        #s = os.path.basename(outdir)
        datadir = f"{outdir}"
        recorddir = f"{datadir}/record"

        inputFasta = f"{datadir}/seq.txt"
        if not os.path.isfile(inputFasta):
            print(f"ERROR: {datadir}/seq.txt missing")
            exit(3)
        Lch = fasta2len(inputFasta)

        if (not os.path.isfile(f"{datadir}/MSA/protein.a3m") and os.path.isfile(f"{datadir}/MSA/dMSA.hhba3m")) or \
                (not os.path.isfile(f"{datadir}/JGI/protein.a3m") and os.path.isfile(f"{datadir}/JGI/q4JGI.a3m")):
            
            cpfinal = "no"
            tag = f"sMSA_{s.strip('/')}"
            if re.search(f"{tag}", qzy):
                print(f"{tag} job is running!")
                exit(4)

            print("Doing MSA select!\n")
            mem = "15GB"
            cpu = dcpu
            '''
            if Lch > 300:
                mem = "20GB"
                cpu = 6
            if Lch > 800:
                mem = "25GB"
                cpu = 6
            if Lch > 1000:
                mem = "30GB"
                cpu = 8
            '''
            if Lch >= 100:
                cpfinal = "sbatch"
            jobname = f"{recorddir}/{tag}"

            cmdheader = command_header(tag=tag, jobname=jobname, partition=partition, mem=mem, cpu=cpu, account=account)
            cmdcontent = f"{python3} {bindir}/MSA_JGI_select.py {datadir} /tmp/{os.environ['USER']}/{tag} {run_type}\n"

            jobcontent = f"{cmdheader}\n\n{cmdcontent}\n"

            with open(jobname, "w") as fp:
                fp.write(jobcontent)
            os.system(f"chmod a+x {jobname}")
            #print(run_type)
            bsub = submitjob(jobname, run_type)

            date = subprocess.run(["date"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip('\n')
            with open(f"{recorddir}/note.txt", "a") as f:
                print(f"{jobname}\t at {date} {bsub}")
                f.write(f"{jobname}\t at {date} {bsub}\n")

        else:
            if os.path.isfile(f"{datadir}/JGI/q4JGI.a3m") and os.path.isfile(f"{datadir}/JGI/protein.a3m"):
                print("JGI/protein.a3m exists, skip MSA select!\n")

            elif os.path.isfile(f"{datadir}/MSA/protein.a3m") and os.path.isfile(f"{datadir}/MSA/dMSA.hhba3m"):
                print("MSA/protein.a3m exists, skip MSA select!\n")
            else:
                print("No MSA, skip MSA select!\n")

    elif method_arg == "alphafold2":
        print("Start doing MSAs ranking by AlphaFold2\n")
        cmd = f"{python3} {bindir}/run_AF2_multiMSA_sub.py -o={outdir} -run_type={run_type}"
        content = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")
        print(content)

    if os.path.exists(os.path.join(outdir,"MSA/protein.a3m")) and os.path.exists(os.path.join(outdir,"MSA/protein.aln")) and os.path.exists(os.path.join(outdir,"finalMSAs/MSA_ranking.info")):
        print("MSA ranking for %s ended normally!\n"%s)
