import argparse
import os
import subprocess
import time
from lib.tool import tool_utils
from lib.utils.execute import execute

para_json = dict(
    
    # database parameter 
    dMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'uniclust30_2017_04/uniclust30_2017_04'),
    dMSAjackhmmerdb=os.path.join("/data/protein/datasets_2022", 'uniref90/uniref90.fasta'),
    dMSAhmmsearchdb=os.path.join("/data/protein/datasets_2024", 'metaclust/metaclust.fasta'),
    qMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'UniRef30_2302'),
    qMSAjackhmmerdb=os.path.join("/data/protein/datasets_2022", 'uniref90/uniref90.fasta'),
    qMSAhhblits3db=os.path.join("/data/protein/alphafold", 'bfd'),
    qMSAhmmsearchdb=os.path.join("/data/protein/datasets_2022", 'mgnify/mgy_clusters.fa'),
    mMSAJGI=os.path.join("/data/protein/datasets_2024", 'JGIclust')
)


def submitjob(jobname):
    bsub = ""
    while len(bsub) == 0:
        # run under standalone machine
        bsub = subprocess.run(["bash", jobname], stdout=subprocess.PIPE).stdout.decode("utf-8")
        bsub = bsub.strip("\n")
        if len(bsub):
            break

        time.sleep(60)
    return bsub


def img_main(args):
    #
    missing = 0
    JGI = para_json['mMSAJGI']
    HHLIB = f"{para_json['qMSApkg']}"
    hhblitsdb = para_json['qMSAhhblitsdb']
    jackhmmerdb = para_json['qMSAjackhmmerdb']
    hhblits3db = para_json['qMSAhhblits3db']

    content = subprocess.run(["cat", f"{JGI}/list"], stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")
    for j in range(len(content)):
            DBfasta = content[j].strip('\n')
            if DBfasta == "":
                break
            if os.path.exists(f"{args.datadir}/JGI/{DBfasta}.cdhit"):
                continue
            
            missing += 1
            with tool_utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
                deepm_path = os.path.join(query_tmp_dir, "deepm.sh")
                cmd_header = f"#!/bin/bash\n"

                cmd_content = f"cp {args.datadir}/MSA/qMSA.hh3aln seq.hh3aln\n" \
                            f"if [ ! -s 'seq.hh3aln' ];then\n" \
                            f"    cp {args.datadir}/MSA/dMSA.jacaln seq.hh3aln\n" \
                            f"fi\n" \
                            f"if [ ! -s 'seq.hh3aln' ];then\n" \
                            f"    cp {args.datadir}/MSA/dMSA.hhbaln seq.hh3aln\n" \
                            f"fi\n\n" \
                            f"sed = seq.hh3aln |sed 'N;s/\\n/\\t/'|sed 's/^/>/g'|sed 's/\\t/\\n/g'| {HHLIB}/bin/qhmmbuild -n aln --amino -O seq.afq --informat afa seq.hmm -\n\n" \
                            f"{HHLIB}/bin/qhmmsearch --cpu 1 -E 10 --incE 1e-3 -A {DBfasta}.match --tblout {DBfasta}.tbl -o {DBfasta}.out seq.hmm {JGI}/{DBfasta}\n" \
                            f"{HHLIB}/bin/esl-sfetch -f {JGI}/{DBfasta} {DBfasta}.tbl|sed 's/*//g' > {DBfasta}.fseqs\n" \
                            f"{HHLIB}/bin/cd-hit -i {DBfasta}.fseqs -o {args.datadir}/JGI/{DBfasta}.cdhit -c 1 -M 3000\n\n"

                job_content = f"{cmd_header}\n\n{cmd_content}\n"

                with open(deepm_path, "w") as fp:
                    fp.write(job_content)

                os.system(f"chmod a+x {deepm_path}")
                submitjob(deepm_path)

    
    if missing > 0:
        print(f"{missing} JGI search still running. Skip combination")
        exit(5)
    
    if os.path.exists(f"{args.datadir}/JGI/DeepJGI.a3m") and os.path.exists(f"{args.datadir}/JGI/q3JGI.a3m") and os.path.exists(f"{args.datadir}/JGI/q4JGI.a3m"):
        print(f"DeepMSA2_IMG is finished.")
        exit(6)
    
    print("2rd step/JGI combination is starting!\n")
    Q = "urgent"
    if Q == "default":
        Q = "normal"

    comb_cmd = f"python JGImod.py {args.datadir} /tmp {hhblitsdb} {jackhmmerdb} {hhblits3db} {Q}\n"
    execute(" ".join(comb_cmd))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blast_type", type=str, default="psiblast")
    parser.add_argument("-db", "--database", type=str, default="")

    args = parser.parse_args()

    img_main(args)