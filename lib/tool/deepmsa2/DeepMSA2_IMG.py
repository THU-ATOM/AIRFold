#!/usr/bin/env python3
import os
import sys
import re
import subprocess
from config import para_json
from utils import job_queue, submitjob, command_header, fasta2len

docstring = '''
DeepMSA2_IMG.py 

    MSA construction by mMSA, it will use JGIclust, MetaSourceDB and TaraDB databases.
    you may need run two times of this script to get the final results.

Usage: DeepMSA2_IMG.py [option]

    required options:
    -i=/home/simth/test/seq.fasta
    -o=/home/simth/test (This should be the same output directory with DeepMSA2_noIMG.py step)

    optional options:
    -run_type=local (default) or sbatch

'''

if __name__ == "__main__":

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
        run_type = 1 ### local

    # job parameter
    HHLIB = f"{para_json['qMSApkg']}"
    DMSALIB = f"{para_json['dMSApkg']}"
    cpu = para_json['mMSAcpu']
    account = para_json['account']
    partition = para_json['partition']
    Q = "urgent"
    if Q == "default":
        Q = "normal"
    JGI = para_json['mMSAJGI']
    hhblitsdb = para_json['qMSAhhblitsdb']
    jackhmmerdb = para_json['qMSAjackhmmerdb']
    hhblits3db = para_json['qMSAhhblits3db']
    hmmsearchdb = para_json['qMSAhmmsearchdb']
    python3=para_json['python_DeepPotential']
    ##### job start #########
    qzy = job_queue(run_type)

    s = os.path.basename(outdir)
    datadir = f"{outdir}"
    recorddir = f"{datadir}/record"

    bsub = ""
    Lch = fasta2len(f"{datadir}/MSA/qMSA.fasta")
    if Lch > 300:
        mem = '15GB'
    else:
        mem = '10GB'

    if not os.path.exists(recorddir):
        os.system(f"mkdir -p {recorddir}")
    if not os.path.exists(f"{datadir}/JGI"):
        os.system(f"mkdir -p {datadir}/JGI")

    # search JGI
    if not os.path.isfile(f"{datadir}/MSA/qMSA.a3m"):
        print(f"{s} does not have MSA result yet. Please run DeepMSA2_noIMG.py first!")
        exit(3)
    elif not os.path.isfile(f"{datadir}/MSA/qMSA.jaca3m") and not os.path.isfile(f"{datadir}/MSA/dMSA.jaca3m"):
        print(f"{s} does not need additional JGI search due to no jack result. skip")
        exit(4)    

    missing = 0
    print("1st step is starting!\n")
    content = subprocess.run(["cat", f"{JGI}/list"], stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")
    for j in range(len(content)):
        DBfasta = content[j].strip('\n')
        if DBfasta == "":
            break
        if os.path.exists(f"{datadir}/JGI/{DBfasta}.cdhit"):
            continue
        missing += 1
        tag = f"{DBfasta}_{s.strip('/')}"
        jobname = f"{datadir}/record/{tag}"
        user_id = os.environ["USER"]
        tmpdir = f"/tmp/{user_id}/{tag}"

        cmdheader = command_header(tag=tag, jobname=jobname, partition=partition, mem=mem, cpu=cpu, account=account)

        cmdcontent = f"mkdir -p {tmpdir}\n" \
                     f"cd {tmpdir}\n\n" \
                     f"echo hostname: `hostname`  >>{recorddir}/ware_{tag}\n" \
                     f"echo starting time: `date` >>{recorddir}/ware_{tag}\n" \
                     f"echo pwd `pwd`             >>{recorddir}/ware_{tag}\n\n" \
                     f"cp {datadir}/MSA/qMSA.hh3aln seq.hh3aln\n" \
                     f"if [ ! -s 'seq.hh3aln' ];then\n" \
                     f"    cp {datadir}/MSA/dMSA.jacaln seq.hh3aln\n" \
                     f"fi\n" \
                     f"if [ ! -s 'seq.hh3aln' ];then\n" \
                     f"    cp {datadir}/MSA/dMSA.hhbaln seq.hh3aln\n" \
                     f"fi\n\n" \
                     f"sed = seq.hh3aln |sed 'N;s/\\n/\\t/'|sed 's/^/>/g'|sed 's/\\t/\\n/g'| {HHLIB}/bin/qhmmbuild -n aln --amino -O seq.afq --informat afa seq.hmm -\n\n" \
                     f"{HHLIB}/bin/qhmmsearch --cpu 1 -E 10 --incE 1e-3 -A {DBfasta}.match --tblout {DBfasta}.tbl -o {DBfasta}.out seq.hmm {JGI}/{DBfasta}\n" \
                     f"{HHLIB}/bin/esl-sfetch -f {JGI}/{DBfasta} {DBfasta}.tbl|sed 's/*//g' > {DBfasta}.fseqs\n" \
                     f"{HHLIB}/bin/cd-hit -i {DBfasta}.fseqs -o {datadir}/JGI/{DBfasta}.cdhit -c 1 -M 3000\n\n" \
                     f"sync\n" \
                     f"rm -rf {tmpdir}\n" \
                     f"echo ending time: `date`   >>{recorddir}/ware_{tag}\n"

        jobcontent = f"{cmdheader}\n\n{cmdcontent}\n"

        with open(jobname, "w") as fp:
            fp.write(jobcontent)

        os.system(f"chmod a+x {jobname}")

        qzy = job_queue(run_type)
        test_job = re.search(f"{tag}", qzy)
        if test_job:
            print(f"{tag} was submitted. skip")
        else:
            bsub = submitjob(jobname, run_type)

        date = subprocess.run(["date"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip('\n')
        with open(f"{recorddir}/note.txt", "a") as f:
            print(f"{jobname}\t at {date} {bsub}")
            f.write(f"{jobname}\t at {date} {bsub}\n")

    

    tag = f"sJGI_{s.strip('/')}"
    if missing > 0:
        print(f"{missing} JGI search still running for {s}. Skip combination")
        exit(5)
    if os.path.exists(f"{datadir}/JGI/DeepJGI.a3m") and os.path.exists(f"{datadir}/JGI/q3JGI.a3m") and os.path.exists(f"{datadir}/JGI/q4JGI.a3m"):
        print(f"DeepMSA2_IMG for {s} is finished.")
        exit(6)
    print("2rd step/JGI combination is starting!\n")
    jobname = f"{datadir}/record/{tag}"

    cmdheader = command_header(tag=tag, jobname=jobname, partition=partition, mem=mem, cpu=cpu, account=account)

    cmdcontent = f"mkdir -p /tmp/{user}/{tag}\n" \
                 f"{python3} {bindir}/JGImod.py {datadir} /tmp/{user}/{tag} {hhblitsdb} {jackhmmerdb} {hhblits3db} {Q}\n" \

    jobcontent = f"{cmdheader}\n\n{cmdcontent}\n"

    with open(jobname, "w") as fp:
        fp.write(jobcontent)

    os.system(f"chmod a+x {jobname}")

    qzy = job_queue(run_type)
    test_job = re.search(f"{tag}", qzy)
    if test_job:
        print(f"{tag} submitted. skip")
    else:
        bsub = submitjob(jobname, run_type)

        date = subprocess.run(["date"], stdout=subprocess.PIPE).stdout.decode("utf-8").strip('\n')
        with open(f"{recorddir}/note.txt", "a") as f:
            print(f"{jobname}\t at {date} {bsub}")
            f.write(f"{jobname}\t at {date} {bsub}\n")




