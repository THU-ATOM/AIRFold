import os
import time
import subprocess

def fasta2len(filename):
    sequence = ""
    seqfile=open(filename,'r')
    seqlines=seqfile.readlines()
    seqfile.close()
    for seqline in seqlines:
        if not seqline.startswith(">"):
            sequence+=seqline.strip("\n")
    Lch = len(sequence)
    return Lch

def read_sequence_from_fasta(seq_txt="seq.fasta"):
    '''read single sequence file "seq_txt". return header and sequence'''
    target_names=[]
    sequences=[]
    fp=open(seq_txt)

    text='\n'+fp.read()
    blocks=text.split('\n>')[1:]
    #print len(blocks)
    for block in blocks:
        tmpstrs=block.split('\n')
        target_names.append(tmpstrs[0])
        seq=''
        for tmpstr in tmpstrs[1:]:
            seq+=tmpstr
        sequences.append(seq)
    fp.close()
    return target_names,sequences


def command_header(tag, jobname, partition, mem, cpu, account):
    cmd_header = f"#!/bin/bash\n"\
                 f"#SBATCH --job-name={tag}\n" \
                 f"#SBATCH --output={jobname}.out\n" \
                 f"#SBATCH --error={jobname}.err\n" \
                 f"#SBATCH --partition={partition}\n" \
                 f"#SBATCH --nodes=1\n" \
                 f"#SBATCH --mem={mem}\n" \
                 f"#SBATCH --ntasks-per-node={cpu}\n" \
                 f"#SBATCH --export=ALL\n" \
                 f"#SBATCH -t 24:00:00\n" \
                 f"#SBATCH --account={account}\n"
    return cmd_header


def command_header_gpu(tag, jobname, partition, mem, cpu, account):
    cmd_header = f"#!/bin/bash\n"\
                 f"#SBATCH --job-name={tag}\n" \
                 f"#SBATCH --output={jobname}.out\n" \
                 f"#SBATCH --error={jobname}.err\n" \
                 f"#SBATCH --partition={partition}\n" \
                 f"#SBATCH --nodes=1\n" \
                 f"#SBATCH --mem={mem}\n" \
                 f"#SBATCH --ntasks-per-node={cpu}\n" \
                 f"#SBATCH --export=ALL\n" \
                 f"#SBATCH -t 24:00:00\n" \
                 f"#SBATCH --account={account}\n" \
                 f"#SBATCH --gpus=1\n\n" \
                 f"use_gpu='true'\n\n" #\
                 #f"if [ '$use_gpu' == 'true' ]\n" \
                 #f"then\n" \
                 #f"    #module purge\n" \
                 #f"    #module load gpu\n" \
                 #f"    #module load slurm\n" \
                 #f"    #module load openmpi\n" \
                 #f"    module load cudnn/11.5-v8.3.1\n" \
                 #f"    module load cuda/11.5.1\n" \
                 #f"fi\n"

    return cmd_header


def job_queue(run_type):
    user_id = os.environ["USER"]

    if run_type == 2:
        # run under slurm system
        qzy = subprocess.run(["squeue", '-u', user_id, "-o", "%j"], stdout=subprocess.PIPE).stdout.decode("utf-8")
    else:
        # run under standalone machine
        cmd = f"ps -u {user_id} -c"
        qzy = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")

    return qzy


def submitjob(jobname, run_type):
    bsub = ""
    while len(bsub) == 0:
        if run_type == 2:
            # run under slurm
            bsub = subprocess.run(["sbatch", jobname], stdout=subprocess.PIPE).stdout.decode("utf-8")
        else:
            # run under standalone machine
            bsub = subprocess.run(["bash", jobname], stdout=subprocess.PIPE).stdout.decode("utf-8")

        bsub = bsub.strip("\n")

        if len(bsub):
            break

        time.sleep(60)

    return bsub
