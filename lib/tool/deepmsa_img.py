import argparse
import os
import re
import subprocess
import time

rootpath = "/home/casp15/code/MSA/DeepMSA2"
databasesrootpath = "/data/protein/datasets_2024"

para_json = dict(
    # main program parameter
    programrootpath=rootpath,
    
    qMSApkg=os.path.join(rootpath, "bin/qMSA"),
    dMSApkg=os.path.join(rootpath, "bin/dMSA"),
    python_DeepPotential=os.path.join(rootpath, "anaconda3/bin/python"),

    # submit job parameter
    run_type='local',  # 'local' or 'sbatch'
    partition='xxx_cpu',
    gpu_partition='xxx_gpu',
    account='xxx',
    mMSAcpu=10,
    qMSAcpu=10,
    dMSAcpu=10,

    # database parameter 
    # If you modified the following databases with different version
    # please go to the alphafold and alphafold_multimer folders in bin folder
    # change the corresponding databases in run_alphafold_*.sh
    dMSAhhblitsdb=os.path.join(databasesrootpath, 'uniclust30_2017_04/uniclust30_2017_04'),
    dMSAjackhmmerdb=os.path.join(databasesrootpath, 'uniref90/uniref90.fasta'),
    dMSAhmmsearchdb=os.path.join(databasesrootpath, 'metaclust/metaclust.fasta'),
    qMSAhhblitsdb=os.path.join(databasesrootpath, 'UniRef30_2022_02/UniRef30_2022_02'),
    qMSAjackhmmerdb=os.path.join(databasesrootpath, 'uniref90/uniref90.fasta'),
    qMSAhhblits3db=os.path.join(databasesrootpath, 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'),
    qMSAhmmsearchdb=os.path.join(databasesrootpath, 'mgnify/mgy_clusters.fasta'),
    mMSAJGI=os.path.join(databasesrootpath, 'JGIclust')
)

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


def img_main(args):
    run_type = 1
    JGI = para_json['mMSAJGI']
    HHLIB = f"{para_json['qMSApkg']}"
    DMSALIB = f"{para_json['dMSApkg']}"
    cpu = para_json['mMSAcpu']
    account = para_json['account']
    partition = para_json['partition']
    
    Lch = fasta2len(f"{args.input_fasta}")
    if Lch > 300:
        mem = '15GB'
    else:
        mem = '10GB'
    
    s = os.path.basename(args.outdir)
    recorddir = f"{args.datadir}/record"
    content = subprocess.run(["cat", f"{JGI}/list"], stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")
    for j in range(len(content)):
            DBfasta = content[j].strip('\n')
            if DBfasta == "":
                break
            if os.path.exists(f"{args.datadir}/JGI/{DBfasta}.cdhit"):
                continue
            missing += 1
            tag = f"{DBfasta}_{s.strip('/')}"
            jobname = f"{args.datadir}/record/{tag}"
            user_id = os.environ["USER"]
            tmpdir = f"/tmp/{user_id}/{tag}"
            cmdheader = command_header(tag=tag, jobname=jobname, partition=partition, mem=mem, cpu=cpu, account=account)
            cmdcontent = f"mkdir -p {tmpdir}\n" \
                        f"cd {tmpdir}\n\n" \
                        f"echo hostname: `hostname`  >>{recorddir}/ware_{tag}\n" \
                        f"echo starting time: `date` >>{recorddir}/ware_{tag}\n" \
                        f"echo pwd `pwd`             >>{recorddir}/ware_{tag}\n\n" \
                        f"cp {args.datadir}/MSA/qMSA.hh3aln seq.hh3aln\n" \
                        f"if [ ! -s 'seq.hh3aln' ];then\n" \
                        f"    cp {args.datadir}/MSA/dMSA.jacaln seq.hh3aln\n" \
                        f"fi\n" \
                        f"if [ ! -s 'seq.hh3aln' ];then\n" \
                        f"    cp {args.datadir}/MSA/dMSA.hhbaln seq.hh3aln\n" \
                        f"fi\n\n" \
                        f"sed = seq.hh3aln |sed 'N;s/\\n/\\t/'|sed 's/^/>/g'|sed 's/\\t/\\n/g'| {HHLIB}/bin/qhmmbuild -n aln --amino -O seq.afq --informat afa seq.hmm -\n\n" \
                        f"{HHLIB}/bin/qhmmsearch --cpu 1 -E 10 --incE 1e-3 -A {DBfasta}.match --tblout {DBfasta}.tbl -o {DBfasta}.out seq.hmm {JGI}/{DBfasta}\n" \
                        f"{HHLIB}/bin/esl-sfetch -f {JGI}/{DBfasta} {DBfasta}.tbl|sed 's/*//g' > {DBfasta}.fseqs\n" \
                        f"{HHLIB}/bin/cd-hit -i {DBfasta}.fseqs -o {args.datadir}/JGI/{DBfasta}.cdhit -c 1 -M 3000\n\n" \
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blast_type", type=str, default="psiblast")
    parser.add_argument("-db", "--database", type=str, default="")
    parser.add_argument(
        "-of", "--outfmt", type=str, default="6 sseqid  qstart qend qseq sseq"
    )
    parser.add_argument("-cpu", "--threads", type=int, default=64)
    parser.add_argument("-e", "--evalue", type=float, default=1e-3)
    parser.add_argument("-n", "--num_iterations", type=int, default=3)
    parser.add_argument("-i", "--fasta_path", type=str, required=True)
    parser.add_argument("-o", "--a3m_path", type=str, required=True)
    parser.add_argument("-w", "--whole_seq_path", type=str, required=True)

    args = parser.parse_args()

    img_main(args)