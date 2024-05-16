import os
import subprocess
import argparse

para_json = dict(
    # database parameter 
    dMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'uniclust30_2017_04/uniclust30_2017_04'),
    dMSAjackhmmerdb=os.path.join("/data/protein/datasets_2024", 'uniref90/uniref90.fasta'),
    dMSAhmmsearchdb=os.path.join("/data/protein/datasets_2024", 'metaclust/metaclust.fasta'),
    qMSAhhblitsdb=os.path.join("/data/protein/datasets_2024", 'UniRef30_2022_02/UniRef30_2022_02'),
    qMSAjackhmmerdb=os.path.join("/data/protein/datasets_2024", 'uniref90/uniref90.fasta'),
    qMSAhhblits3db=os.path.join("/data/protein/alphafold", 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'),
    qMSAhmmsearchdb=os.path.join("/data/protein/datasets_2022", 'mgnify/mgy_clusters.fa'),
    mMSAJGI=os.path.join("/data/protein/datasets_2024", 'JGIclust')
)


def q34JGI(deepmmsa_base, deepmmsa_base_temp, args):
    
    deepqmsa_hhbaln = args.deepqmsa_hhbaln
    deepqmsa_hhba3m = args.deepqmsa_hhba3m
    deepqmsa_jacaln = args.deepqmsa_jacaln
    deepqmsa_jaca3m = args.deepqmsa_jaca3m
    deepqmsa_hh3aln = args.deepqmsa_hh3aln
    deepqmsa_hh3a3m = args.deepqmsa_hh3a3m
    
    os.system(f"cat {deepqmsa_hhba3m} | {HHLIB}/bin/unaligna3m - {deepmmsa_base_temp}/DB.fasta.fseqs")
    os.system(f"cat {deepmmsa_base}/DB.fasta.*.cdhit >> {deepmmsa_base_temp}/DB.fasta.fseqs")

    cmd = f"{HHLIB}/bin/esl-sfetch --index {deepmmsa_base_temp}/DB.fasta.fseqs"
    print(f"Make index - {cmd}\n")
    content = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")
    print(content)

    os.system(f"ln -s {args.seq} {deepmmsa_base}/q4JGI.fasta")
    os.system(f"ln -s {deepqmsa_hhbaln} {deepmmsa_base}/q4JGI.hhbaln")
    os.system(f"ln -s {deepqmsa_hhba3m} {deepmmsa_base}/q4JGI.hhba3m")
    os.system(f"ln -s {deepqmsa_jacaln} {deepmmsa_base}/q4JGI.jacaln")
    os.system(f"ln -s {deepqmsa_jaca3m} {deepmmsa_base}/q4JGI.jaca3m")
    os.system(f"ln -s {deepqmsa_hh3aln} {deepmmsa_base}/q4JGI.hh3aln")
    os.system(f"ln -s {deepqmsa_hh3a3m} {deepmmsa_base}/q4JGI.hh3a3m")

    cmd = f"python {HHLIB}/scripts/qMSA2.py -hhblitsdb={qhhblitsdb} -jackhmmerdb={jackhmmerdb} -hhblits3db={qhhblits3db} -hmmsearchdb={deepmmsa_base_temp}/DB.fasta.fseqs -tmpdir={deepmmsa_base_temp} {deepmmsa_base}/q4JGI.fasta"

    if not os.path.isfile(f"{deepmmsa_base}/q4JGI.a3m"):
        print(f"first qMSA2 - {cmd}")
        content = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")
        print(content)

    # make q3JGI.{a3m, aln} ##
    os.system(f"ln -s {args.seq} {deepmmsa_base}/q3JGI.fasta")
    os.system(f"ln -s {deepqmsa_hhbaln} {deepmmsa_base}/q3JGI.hhbaln")
    os.system(f"ln -s {deepqmsa_hhba3m} {deepmmsa_base}/q3JGI.hhba3m")
    os.system(f"ln -s {deepqmsa_jacaln} {deepmmsa_base}/q3JGI.jacaln")
    os.system(f"ln -s {deepqmsa_jaca3m} {deepmmsa_base}/q3JGI.jaca3m")

    cmd = f"python {HHLIB}/scripts/qMSA2.py -hhblitsdb={qhhblitsdb} -jackhmmerdb={jackhmmerdb} -hmmsearchdb={deepmmsa_base_temp}/DB.fasta.fseqs -tmpdir={deepmmsa_base_temp} {deepmmsa_base}/q3JGI.fasta"

    if not os.path.isfile(f"{deepmmsa_base}/q3JGI.a3m"):
        print(f"second qMSA2 - {cmd}")
        content = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")
        print(content)


def deepJGI(deepmmsa_base, deepmmsa_base_temp, args):
    deepdmsa_hhbaln = args.deepdmsa_hhbaln
    deepdmsa_hhba3m = args.deepdmsa_hhba3m
    deepdmsa_jacaln = args.deepdmsa_jacaln
    deepdmsa_jaca3m = args.deepqmsa_jaca3m
    os.system(f"ln -s {args.seq} {deepmmsa_base}/DeepJGI.fasta")
    os.system(f"ln -s {deepdmsa_hhbaln} {deepmmsa_base}/DeepJGI.hhbaln")
    os.system(f"ln -s {deepdmsa_hhba3m} {deepmmsa_base}/DeepJGI.hhba3m")
    os.system(f"ln -s {deepdmsa_jacaln} {deepmmsa_base}/DeepJGI.jacaln")
    os.system(f"ln -s {deepdmsa_jaca3m} {deepmmsa_base}/DeepJGI.jaca3m")
    
    cmd = f"python {DMSALIB}/scripts/build_MSA.py -hhblitsdb={dhhblitsdb} -jackhmmerdb={jackhmmerdb} -hmmsearchdb={deepmmsa_base_temp}/DB.fasta.fseqs -tmpdir={deepmmsa_base_temp} {deepmmsa_base}/DeepJGI.fasta"

    if not os.path.isfile(f"{deepmmsa_base}/DeepJGI.a3m"):
        print(f"first dMSA2 - {cmd}")
        content = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True).stdout.decode("utf-8")
        print(content)

    if not os.path.isfile(f"{deepmmsa_base}/DeepJGI.a3m"):
        if os.path.isfile(f"{deepmmsa_base}/DeepJGI.hmsa3m"):
            os.system(f"cp {deepmmsa_base}/DeepJGI.hmsa3m {deepmmsa_base}/DeepJGI.a3m")
        elif os.path.isfile(f"{deepmmsa_base}/DeepJGI.jaca3m"):
            os.system(f"cp {deepmmsa_base}/DeepJGI.jaca3m {deepmmsa_base}/DeepJGI.a3m")
        else:
            os.system(f"cp {deepmmsa_base}/DeepJGI.hhba3m {deepmmsa_base}/DeepJGI.a3m")


def main(args):
    deepmmsa_base = args.deepmmsa_base
    deepmmsa_base_temp = args.deepmmsa_base_temp
    deepqmsa_base_temp = args.deepqmsa_base_temp
    deepdmsa_base_temp = args.deepdmsa_base_temp
    
    # os.system(f"mkdir -p {deepmmsa_base_temp}")

    # qMSA
    q34JGI(deepmmsa_base, deepmmsa_base_temp, args)
    # dMSA
    deepJGI(deepmmsa_base, deepmmsa_base_temp, args)
    

    print(f"tmpdir = {deepmmsa_base_temp}")
    if os.path.exists(deepmmsa_base_temp):
        os.system(f"rm -rf {deepmmsa_base_temp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hhlib", type=str, default="HHLIB")
    parser.add_argument("--dmsalib", type=str, default="DMSALIB")
    parser.add_argument("--deepmmsa_base", type=str, default="/deepmmsa_base")
    parser.add_argument("--deepmmsa_base_temp", type=str, default="/deepmmsa_base/tmp")
    parser.add_argument("--deepqmsa_base_temp", type=str, default="/deepqmsa_base/tmp")
    parser.add_argument("--deepdmsa_base_temp", type=str, default="/deepdmsa_base/tmp")
    parser.add_argument("--seq", type=str, default="seq.fasta")
    # qMSA for q34JGI
    parser.add_argument("--deepqmsa_hhbaln", type=str, default="qMSA.hhbaln")
    parser.add_argument("--deepqmsa_hhba3m", type=str, default="qMSA.hhba3m")
    parser.add_argument("--deepqmsa_jacaln", type=str, default="qMSA.jacaln")
    parser.add_argument("--deepqmsa_jaca3m", type=str, default="qMSA.jaca3m")
    parser.add_argument("--deepqmsa_hh3aln", type=str, default="qMSA.hh3aln")
    parser.add_argument("--deepqmsa_hh3a3m", type=str, default="qMSA.hh3a3m")
    # dMSA for deepJGI
    parser.add_argument("--deepdmsa_hhbaln", type=str, default="dMSA.hhbaln")
    parser.add_argument("--deepdmsa_hhba3m", type=str, default="dMSA.hhba3m")
    parser.add_argument("--deepdmsa_jacaln", type=str, default="dMSA.jacaln")
    parser.add_argument("--deepdmsa_jaca3m", type=str, default="dMSA.jaca3m")

    args = parser.parse_args()

    HHLIB = args.hhlib
    DMSALIB = args.dmsalib
    qhhblitsdb = para_json['qMSAhhblitsdb']
    qhhblits3db = para_json['qMSAhhblits3db']
    dhhblitsdb = para_json['dMSAhhblitsdb']
    jackhmmerdb = para_json['qMSAjackhmmerdb']
    
    main(args)
