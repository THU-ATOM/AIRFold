"""Search MSA using MMseqs.

Output files
----------

- raw_search: hhblits output
- clean_search: only reserve target sequence + retrieved sequences. If the
    input sequence of hhblit is a 


Format
----------
- a3m: homology sequences have insertions (lower characters)
- aln: only contains aligned sequences, no extra comments
- fasta: based on aln, but contains a comment line start with `>` before each sequence

"""
import argparse
import os
import re
import hashlib
import requests
import tarfile
import time
import random
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from lib.constant import TMP_ROOT
import lib.utils.datatool as dtool


def prep_sequence(sequence):
    # this is a fucntion borrow from the cf package
    sequence = str(sequence)
    sequence = re.sub("[^A-Z:/]", "", sequence.upper())
    sequence = re.sub(":+", ":", sequence)
    sequence = re.sub("/+", "/", sequence)
    sequence = re.sub("^[:/]+", "", sequence)
    sequence = re.sub("[:/]+$", "", sequence)

    return sequence


TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"


def run_mmseqs2(
    x,
    prefix,
    use_env=True,
    use_filter=True,
    use_templates=False,
    filter=None,
    host_url="https://a3m.mmseqs.com",
):
    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        res = requests.post(
            f"{host_url}/ticket/msa", data={"q": query, "mode": mode}
        )
        try:
            out = res.json()
        except ValueError:
            out = {"status": "UNKNOWN"}
        return out

    def status(ID):
        res = requests.get(f"{host_url}/ticket/{ID}")
        try:
            out = res.json()
        except ValueError:
            out = {"status": "UNKNOWN"}
        return out

    def download(ID, path):
        res = requests.get(f"{host_url}/result/download/{ID}")
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    # define path
    path = f"{prefix}_{mode}"
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    #   if not os.path.isdir(path):
    #        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = sorted(list(set(seqs)))
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    # resubmit
                    time.sleep(5 + random.randint(0, 5))
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    REDO = False
                    raise Exception(
                        f"MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later."
                    )

                if out["status"] == "MAINTENANCE":
                    REDO = False
                    raise Exception(
                        f"MMseqs2 API is undergoing maintanance. Please try again in a few minutes."
                    )

                # wait for job to finish
                ID, TIME = out["id"], 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)
                    # if TIME > 900 and out["status"] != "COMPLETE":
                    #  # something failed on the server side, need to resubmit
                    #  N += 1
                    #  break

                if out["status"] == "COMPLETE":
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env:
        a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if not os.path.isfile(a3m_files[0]):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # templates
    if use_templates:
        templates = {}
        logger.info("seq\tpdb\tcid\tevalue")
        for line in open(f"{path}/pdb70.m8", "r"):
            p = line.rstrip().split()
            M, pdb, qid, e_value = p[0], p[1], p[2], p[10]
            M = int(M)
            if M not in templates:
                templates[M] = []
            templates[M].append(pdb)
            if len(templates[M]) <= 20:
                logger.info(f"{int(M)-N}\t{pdb}\t{qid}\t{e_value}")

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                os.system(
                    f"curl -s https://a3m-templates.mmseqs.com/template/{TMPL_LINE} | tar xzf - -C {TMPL_PATH}/"
                )
                os.system(
                    f"cp {TMPL_PATH}/pdb70_a3m.ffindex {TMPL_PATH}/pdb70_cs219.ffindex"
                )
                os.system(f"touch {TMPL_PATH}/pdb70_cs219.ffdata")
            template_paths[k] = TMPL_PATH

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    # return results
    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
                logger.info(f"{n-N}\tno_templates_found")
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_

    if isinstance(x, str):
        return (
            (a3m_lines[0], template_paths[0]) if use_templates else a3m_lines[0]
        )
    else:
        return (a3m_lines, template_paths) if use_templates else a3m_lines


def get_hash(x):
    return hashlib.sha1(x.encode()).hexdigest()


def mmseqs2_main(args):
    if Path(args.fasta_output).exists():
        logger.info(f"{args.fasta_output} already exists, skip running mmseqs2")
        return
    else:
        logger.info(f"searching mmseqs2: {args.fasta_path}")
        with open(args.fasta_path) as f:
            msas = f.readlines()
            for l_ in msas:
                l_ = l_.strip()
                if l_[0] != ">":
                    seq_ = prep_sequence(l_)
                    logger.info(seq_)
        prefix = get_hash(seq_)
        prefix = os.path.join(TMP_ROOT, prefix)

        A3M_LINES = run_mmseqs2(
            seq_, prefix, use_filter=True, host_url=args.host_url
        )

        (Path(args.a3m_path).parent).mkdir(parents=True, exist_ok=True)
        with open(args.a3m_path, "w") as f:
            f.write(A3M_LINES)

        if Path(args.a3m_path).exists():
            (Path(args.fasta_output).parent).mkdir(parents=True, exist_ok=True)
            dtool.process_file(
                args.a3m_path,
                args.fasta_output,
                lines_funcs=[dtool.a3m2fasta],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-url", "--host_url", type=str, default="https://a3m.mmseqs.com"
    )
    parser.add_argument("-i", "--fasta_path", type=str, required=True)
    parser.add_argument("-o", "--a3m_path", type=str, required=True)
    parser.add_argument("-fo", "--fasta_output", type=str, required=True)
    args = parser.parse_args()

    mmseqs2_main(args)
