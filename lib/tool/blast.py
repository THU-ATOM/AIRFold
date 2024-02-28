import argparse
import os
from loguru import logger
from pathlib import Path
from typing import List

import lib.utils.datatool as dtool
from lib.constant import BLAST_ROOT
from lib.tool import tool_utils
from lib.utils.execute import execute


def santity_check(lines: List[str]):
    "function used to check the sequence is valid or not"
    seq = lines[1]
    valid_characters = set(
        [
            "L",
            "A",
            "G",
            "V",
            "S",
            "E",
            "R",
            "T",
            "I",
            "D",
            "P",
            "K",
            "Q",
            "N",
            "F",
            "Y",
            "M",
            "H",
            "W",
            "C",
            "X",
            "B",
            "U",
            "Z",
            "O",
            ".",
            "-",
        ]
    )
    santity = all(c in valid_characters for c in seq)
    if not santity:
        logger.warning("Warning: sequence contains invalid characters")
        return False
    else:
        # print("Sequence is valid")
        return True


def parse_blast(blast_file: Path, query_lenth: int):
    """
    function used to parse the blastp/psiblast result, with the output format
    parameter as: -outfmt '6 sseqid  qstart qend qseq sseq'"
    """
    result = []
    with open(blast_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        for l_ in lines:
            if len(l_) == 5:
                align_query = l_[3]
                align_subject = l_[4]
                qstart = int(l_[1])
                qend = int(l_[2])
                if len(align_query) == len(align_subject):
                    l_s = list(align_subject)
                    for j in range(len(l_s)):
                        if align_query[j] == "-":
                            l_s[j] = l_s[j].lower()
                    head = "-" * (qstart - 1)
                    tail = "-" * (query_lenth - qend)
                    result.append(head + "".join(l_s) + tail)
                else:
                    continue
                    # raise ValueError(f"{blast_file}:{l_}")
            else:
                logger.info(l_)
    return result


def _get_whole_seq(blast_file: Path, whole_seq: Path):
    result = []
    with open(blast_file) as f:
        lines = f.readlines()
        lines = [line.strip().split() for line in lines]
        for l_ in lines:
            if len(l_) == 5:
                id = l_[0].split("|")[1]
                with tool_utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
                    out_path = os.path.join(query_tmp_dir, "output.sto")
                    # command += f" -out {out_path}"
                    cmd = [
                        "blastdbcmd",
                        "-db",
                        str(BLAST_ROOT),
                        "-entry",
                        id,
                        "-out",
                        out_path,
                    ]
                    # by default in fasta format
                    # logger.info(" ".join(cmd))
                    execute(" ".join(cmd))
                    with open(out_path) as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]
                        if len(lines) != 0 and santity_check(lines):
                            for l_ in lines:
                                result.append(l_)
                        else:
                            continue
            else:
                continue
    with open(whole_seq, "w") as f:
        for l_ in result:
            f.write(l_ + "\n")


def blast_main(args):
    if not Path(args.a3m_path).exists():
        if args.blast_type == "blastp":
            blast_type = "blastp"
        elif args.blast_type == "psiblast":
            blast_type = "psiblast"

        command = (
            f"{blast_type}"
            f" -query { args.fasta_path}"
            f" -db {args.database}"
            f" -outfmt '{args.outfmt}'"
            f" -num_threads {args.threads}"
        )
        if args.evalue:
            command += f" -evalue {args.evalue}"
        if blast_type == "psiblast":
            command += f" -num_iterations {args.num_iterations}"
        with tool_utils.tmpdir_manager(base_dir="/tmp") as query_tmp_dir:
            out_path = os.path.join(query_tmp_dir, "output.sto")
            command += f" -out {out_path}"
            logger.info(command)
            execute(command)
            query_len = len(dtool.fasta2list(args.fasta_path)[0])
            result = parse_blast(out_path, query_len)
            logger.info(f"BLAST has searched {len(result)} sequence")
            (Path(args.a3m_path).parent).mkdir(parents=True, exist_ok=True)
            (Path(args.whole_seq_path).parent).mkdir(parents=True, exist_ok=True)
            _get_whole_seq(out_path, args.whole_seq_path)
            # dtool.list2fasta(args.a3m_path, result)
            with open(args.a3m_path, "w") as f:
            # with open(output_dir, 'w') as f:
                f.writelines([">src=bl" + "\n" + line + "\n" for line in result])

    else:
        logger.info(f"{args.a3m_path} already exists, skip.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--blast_type", type=str, default="psiblast")
    # parser.add_argument(
    #     "-url", "--host_url", type=str, default="https://a3m.mmseqs.com"
    # )
    parser.add_argument("-db", "--database", type=str, default=str(BLAST_ROOT))
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

    # "-query 2022-05-18_T1122.fasta  -db /data/protein/datasets_2022/blast_dbs/nr/nr -outfmt '6 sseqid  qstart qend qseq sseq' -num_threads 32 -out /tmp/1122_psi.out -evalue 0.001 -num_iterations 3"

    blast_main(args)
