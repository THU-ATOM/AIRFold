import argparse
import os
import subprocess
import time
from lib.tool import tool_utils

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
    JGI = args.jgi
    HHLIB = args.hhlib

    content = subprocess.run(["cat", f"{JGI}/list"], stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")
    for j in range(len(content)):
            DBfasta = content[j].strip('\n')
            if DBfasta == "":
                break
            if os.path.exists(f"{args.deepmmsa_base}/JGI/{DBfasta}.cdhit"):
                continue
            
            with tool_utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
                deepmmsa_path = os.path.join(args.deepmmsa_base, "deepmmsa.sh")
                cmd_header = f"#!/bin/bash\n"

                cmd_content = f"cp {args.dmsa_hhbaln} {args.deepmmsa_base}/seq.hh3aln\n\n" \
                              f"sed = {args.deepmmsa_base}/seq.hh3aln |sed 'N;s/\\n/\\t/'|sed 's/^/>/g'|sed 's/\\t/\\n/g'| {HHLIB}/bin/qhmmbuild -n aln --amino -O seq.afq --informat afa seq.hmm -\n\n" \
                              f"{HHLIB}/bin/qhmmsearch --cpu 1 -E 10 --incE 1e-3 -A {DBfasta}.match --tblout {DBfasta}.tbl -o {DBfasta}.out seq.hmm {JGI}/{DBfasta}\n" \
                              f"{HHLIB}/bin/esl-sfetch -f {JGI}/{DBfasta} {DBfasta}.tbl|sed 's/*//g' > {DBfasta}.fseqs\n" \
                              f"{HHLIB}/bin/cd-hit -i {DBfasta}.fseqs -o {args.deepmmsa_base}/JGI/{DBfasta}.cdhit -c 1 -M 3000\n\n"

                job_content = f"{cmd_header}\n\n{cmd_content}\n"

                with open(deepmmsa_path, "w") as fp:
                    fp.write(job_content)

                os.system(f"chmod a+x {deepmmsa_path}")
                submitjob(deepmmsa_path)
    
    # if os.path.exists(f"{args.deepmmsa_base}/JGI/DeepJGI.a3m") and os.path.exists(f"{args.deepmmsa_base}/JGI/q3JGI.a3m") and os.path.exists(f"{args.mmsa_base}/JGI/q4JGI.a3m"):
    #     print(f"DeepMSA2_IMG is finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jgi", type=str, default="JGI")
    parser.add_argument("-h", "--hhlib", type=str, default="HHLIB")
    parser.add_argument("-b", "--deepmmsa_base", type=str, default="/deepmmsa_base")
    parser.add_argument("-d", "--dmsa_hhbaln", type=str, default="dmsa.hhbaln")

    args = parser.parse_args()

    img_main(args)