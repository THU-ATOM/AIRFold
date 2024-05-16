import argparse
import os
import subprocess
import time

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

def mkdir_if_not_exist(tmpdir):
    ''' create folder if not exists '''
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
        
def img_main(args):
    JGI = args.jgi
    HHLIB = args.hhlib
    
    if args.deepmmsa_base and args.deepmmsa_base != '.':
        mkdir_if_not_exist(args.deepmmsa_base)
        mkdir_if_not_exist(args.deepmmsa_base_temp)
        
    content = subprocess.run(["cat", f"{JGI}/list"], stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")
    for j in range(len(content)):
            DBfasta = content[j].strip('\n')
            if DBfasta == "":
                break
            if os.path.exists(f"{args.deepmmsa_base}/{DBfasta}.cdhit"):
                continue

            deepmmsa_path = os.path.join(args.deepmmsa_base, "deepmmsa.sh")
            cmd_header = f"#!/bin/bash\n"

            cmd_content = f"cp {args.deepqmsa_hh3aln} {args.deepmmsa_base}/seq.hh3aln\n" \
                          f"if [ ! -s '{args.deepmmsa_base}/seq.hh3aln' ];then\n" \
                          f"    cp {args.deepqmsa_jacaln} {args.deepmmsa_base}/seq.hh3aln\n" \
                          f"fi\n" \
                          f"if [ ! -s '{args.deepmmsa_base}/seq.hh3aln' ];then\n" \
                          f"    cp {args.deepqmsa_hhbaln} {args.deepmmsa_base}/seq.hh3aln\n" \
                          f"fi\n\n" \
                          f"sed = {args.deepmmsa_base}/seq.hh3aln |sed 'N;s/\\n/\\t/'|sed 's/^/>/g'|sed 's/\\t/\\n/g'| {HHLIB}/bin/qhmmbuild -n aln --amino -O {args.deepmmsa_base_temp}/seq.afq --informat afa {args.deepmmsa_base_temp}/seq.hmm -\n\n" \
                          f"{HHLIB}/bin/qhmmsearch --cpu 4 -E 10 --incE 1e-3 -A {args.deepmmsa_base_temp}/{DBfasta}.match --tblout {args.deepmmsa_base_temp}/{DBfasta}.tbl -o {args.deepmmsa_base_temp}/{DBfasta}.out {args.deepmmsa_base_temp}/seq.hmm {JGI}/{DBfasta}\n" \
                          f"{HHLIB}/bin/esl-sfetch -f {JGI}/{DBfasta} {args.deepmmsa_base_temp}/{DBfasta}.tbl|sed 's/*//g' > {args.deepmmsa_base_temp}/{DBfasta}.fseqs\n" \
                          f"{HHLIB}/bin/cd-hit -i {args.deepmmsa_base_temp}/{DBfasta}.fseqs -o {args.deepmmsa_base}/{DBfasta}.cdhit -c 1 -M 3000\n\n"

            job_content = f"{cmd_header}\n\n{cmd_content}\n"

            with open(deepmmsa_path, "w") as fp:
                fp.write(job_content)

            os.system(f"chmod a+x {deepmmsa_path}")
            submitjob(deepmmsa_path)
    
    # print(f"tmpdir = {args.deepmmsa_base_temp}")
    # if os.path.exists(args.deepmmsa_base_temp):
    #     os.system(f"rm -rf {args.deepmmsa_base_temp}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jgi", type=str, default="JGI")
    parser.add_argument("--hhlib", type=str, default="HHLIB")
    parser.add_argument("--deepmmsa_base", type=str, default="/deepmmsa_base")
    parser.add_argument("--deepmmsa_base_temp", type=str, default="/deepmmsa_base/tmp")
    parser.add_argument("--deepqmsa_hh3aln", type=str, default="deeqmsa.hh3aln")
    parser.add_argument("--deepqmsa_jacaln", type=str, default="deeqmsa.jacaln")
    parser.add_argument("--deepqmsa_hhbaln", type=str, default="deeqmsa.hhbaln")

    args = parser.parse_args()

    img_main(args)