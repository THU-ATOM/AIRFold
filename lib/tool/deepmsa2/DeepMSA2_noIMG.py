#!/usr/bin/env python
import os
import sys
import re
import subprocess
from config import para_json
from utils import job_queue, submitjob, command_header, fasta2len

docstring = '''
DeepMSA2_noIMG.py 

    MSA construction by aMSA, dMSA and qMSA, it will use uniclust30, UniRef30, 
    uniref90, BFD, metaclust and MGnify databases.
    
Usage: 
    DeepMSA2_noIMG.py [option]

    required options:
    -i=/home/simth/test/seq.fasta
    -o=/home/simth/test

    optional options:
    -run_type=local (default) or sbatch


'''


if __name__ == "__main__":

    ######### input setting
    run_type_arg = ""
    inputsourcefile = ""
    outputdirectory = ""

    argv = []
    for arg in sys.argv[1:]:
        if arg.startswith("-i="):
            inputsourcefile = os.path.abspath(arg[len("-i="):])
        elif arg.startswith("-o="):
            outputdirectory = os.path.abspath(arg[len("-o="):])
        elif arg.startswith("-run_type="):
            run_type_arg = arg[len("-run_type="):]
        elif arg.startswith('-'):
            sys.stderr.write("ERROR! No such option %s\n" % arg)
            exit()
        else:
            argv.append(arg)

    #print(len(argv))
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
            os.system(f"mkdir -p {outdir}")
            if not os.path.exists(os.path.join(outdir,"seq.txt")):
                os.system(f"cp -f {inputsourcefile} {outdir}/seq.txt")
            if not os.path.exists(os.path.join(outdir,"seq.fasta")):
                os.system(f"cp -f {inputsourcefile} {outdir}/seq.fasta")
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
        run_type = 1  #local

    ##### sbatch setting ####
    partition = para_json['partition']
    account = para_json['account']
    HHLIB = f"{para_json['qMSApkg']}"
    DMSALIB = f"{para_json['dMSApkg']}"
    dMSAcpu = para_json['dMSAcpu']
    qMSAcpu = para_json['qMSAcpu']
    python3=para_json['python_DeepPotential']
    pkgdir = para_json['alphafold_pkgdir']  ##### af2 package dir
    libdir = para_json['alphafold_libdir']  ##### af2 library dir
    envdir = para_json['alphafold_env']

    temp_dir = os.path.abspath(outdir)
    outdir = os.path.abspath(outdir)

    Q = "urgent"
    if Q == "default":
        Q = "normal"


    ######## job start ######
    qzy = job_queue(run_type)
    
    s = os.path.basename(outdir)
    datadir = f"{outdir}"
    recorddir = f"{datadir}/record"
    if not os.path.exists(recorddir):
        os.system(f"mkdir -p {recorddir}")
    if not os.path.exists(f"{datadir}/MSA"):
        os.system(f"mkdir -p {datadir}/MSA")

    if not os.path.isfile(f"{datadir}/seq.txt"):
        if os.path.isfile(f"{datadir}/seq.fasta"):
            os.system(f"cp {datadir}/seq.fasta {datadir}/seq.txt")
        else:
            print(f"ERROR: {datadir}/seq.txt missing.")
    else:
        os.system(f"cp {datadir}/seq.txt {datadir}/seq.fasta")

    inputFasta = f"{datadir}/seq.txt"
    if not os.path.isfile(inputFasta):
        inputFasta = f"{datadir}/seq.fasta"
        if not os.path.isfile(inputFasta):
            print(f"ERROR: {datadir}/seq.txt missing")

    if not os.path.isfile(f"{datadir}/MSA/dMSA.fasta"):
        os.system(f"cp {inputFasta} {datadir}/MSA/dMSA.fasta")
    if not os.path.isfile(f"{datadir}/MSA/qMSA.fasta"):
        os.system(f"cp {inputFasta} {datadir}/MSA/qMSA.fasta")

    Lch = fasta2len(inputFasta)

    # qMSA
    tag = f"qMSA_{s.strip('/')}"

    if os.path.isfile(f"{datadir}/MSA/qMSA.aln"):
        print("qMSA for %s has been complete!"%s)
    elif re.search(f"{tag}", qzy):
        print("qMSA for %s is running!"%s)
    else: 
        jobname = f"{recorddir}/{tag}"
        mem = "15GB"
        cpu = qMSAcpu
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
        cmdheader = command_header(tag=tag, jobname=jobname, partition=partition, mem=mem, cpu=cpu,
                                   account=account)
        cmdcontent = f"{python3} {HHLIB}/scripts/qMSA2.py " \
                     f"-hhblitsdb={para_json['qMSAhhblitsdb']} " \
                     f"-jackhmmerdb={para_json['qMSAjackhmmerdb']} " \
                     f"-hhblits3db={para_json['qMSAhhblits3db']} " \
                     f"-hmmsearchdb={para_json['qMSAhmmsearchdb']} " \
                     f"-ncpu={cpu} " \
                     f"{datadir}/MSA/qMSA.fasta"

        jobcontent = f"{cmdheader}\n\n{cmdcontent}\n"

        with open(jobname, "w") as fp:
            fp.write(jobcontent)
        os.system(f"chmod a+x {jobname}")
        bsub = submitjob(jobname, run_type)

        date = subprocess.run(["date"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip('\n')
        with open(f"{recorddir}/note.txt", "a") as f:
            print(f"{jobname}\t at {date} {bsub}")
            f.write(f"{jobname}\t at {date} {bsub}\n")

    # dMSA
    tag = f"dMSA_{s.strip('/')}"
    if os.path.isfile(f"{datadir}/MSA/dMSA.aln"):
        print("dMSA for %s has been complete!"%s)
    elif re.search(f"{tag}", qzy):
        print("dMSA for %s is running!"%s)
    else:
        jobname = f"{recorddir}/{tag}"
        mem = "15GB"
        cpu = dMSAcpu
        '''
        if Lch > 400:
            mem = "20GB"
            cpu = 6
        if Lch > 800:
            mem = "30GB"
            cpu = 6
        if Lch > 1000:
            mem = "35GB"
            cpu = 8
        '''
        cmdheader = command_header(tag=tag, jobname=jobname, partition=partition, mem=mem, cpu=cpu,
                                   account=account)
        cmdcontent = f"{python3} {DMSALIB}/scripts/build_MSA.py " \
                     f"-hhblitsdb={para_json['dMSAhhblitsdb']} " \
                     f"-jackhmmerdb={para_json['dMSAjackhmmerdb']} " \
                     f"-hmmsearchdb={para_json['dMSAhmmsearchdb']} " \
                     f"-ncpu={cpu} " \
                     f"{datadir}/MSA/dMSA.fasta"

        jobcontent = f"{cmdheader}\n\n{cmdcontent}\n"

        with open(jobname, "w") as fp:
            fp.write(jobcontent)
        os.system(f"chmod a+x {jobname}")
        bsub = submitjob(jobname, run_type)

        date = subprocess.run(["date"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip('\n')
        with open(f"{recorddir}/note.txt", "a") as f:
            print(f"{jobname}\t at {date} {bsub}")
            f.write(f"{jobname}\t at {date} {bsub}\n")


    if os.path.isfile(f"{datadir}/MSA/dMSA.a3m") and os.path.isfile(f"{datadir}/MSA/qMSA.a3m"):
        print("DeepMSA2_noIMG has been complete for %s!"%s)
