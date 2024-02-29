import argparse
import sys
import shutil
from pathlib import Path
import traceback
from tqdm import tqdm
from lib.utils.execute import execute
from lib.utils import datatool as dtool

# def cluster(path_a3m, path_result, seq_id=0.3, cov_id=0.8):
def _run(a3m_dir, strategy_dir, seq_id, cov_id, rm_tmp_files=False):
    result_root = Path(strategy_dir).parent
    name = Path(strategy_dir).stem
    dealign_root = result_root / "raw_search_dealign_fasta"
    mmseqs2_root = result_root / "mmseqs2"
    mmseqs2_root.mkdir(exist_ok=True, parents=True)
    # name = sample["name"].split(".")[0]
    input_path = Path(a3m_dir)
    if input_path.exists():
        dealign_output_path = dealign_root / f"{name}.fasta"
        dtool.process_file(
            input_path,
            dealign_output_path,
            lines_funcs=[
                dtool.dealign_a3m,
                dtool.append_number,
            ],
        )

    try:
        cmd = (
            f"mmseqs easy-cluster"
            f" {dealign_output_path}"
            f" {mmseqs2_root / name} {mmseqs2_root / 'tmp'}"
            f" --cluster-reassign"
            f" --min-seq-id {seq_id}"
            f" -c {cov_id}"
            # f" --single-step-clustering"
        )
        execute(cmd, print_off=True)

        raw_lines = dtool.read_lines(input_path, strip=True)
        cluster_repr_path = mmseqs2_root / f"{name}_rep_seq.fasta"
        repr_lines = dtool.read_lines(cluster_repr_path, strip=True)

        repr_a3m_lines = [raw_lines[0], raw_lines[1]]
        for line in repr_lines:
            if dtool.is_comment_line(line):
                line_num = int(line.strip().split("=")[-1])
                repr_a3m_lines.extend([raw_lines[line_num], raw_lines[line_num + 1]])

        output_path = result_root / f"{name}.fasta"
        dtool.write_lines(output_path, dtool.a3m2fasta(repr_a3m_lines))
        print(
            f"{name} before: {len(raw_lines) // 2} "
            f"after: {len(repr_a3m_lines) // 2}"
        )
    except:
        print("==========")
        print(f"failed: {cmd}")
        traceback.print_exception(*sys.exc_info())
        print("==========")

    if rm_tmp_files:
        shutil.rmtree(dealign_root)
        shutil.rmtree(mmseqs2_root)


def cluster_list(
    path_list, a3m_root, result_root, rm_tmp_files=False, seq_id=0.3, cov_id=0.8
):
    samples = dtool.read_jsonlines(path_list)
    result_root = Path(result_root)
    dealign_root = result_root / "raw_search_dealign_fasta"
    mmseqs2_root = result_root / "mmseqs2"
    mmseqs2_root.mkdir(exist_ok=True, parents=True)

    for sample in tqdm(samples, ncols=80):
        name = sample["name"].split(".")[0]
        input_path = Path(a3m_root) / f"{name}.a3m"
        if input_path.exists():
            dealign_output_path = dealign_root / f"{name}.fasta"
            dtool.process_file(
                input_path,
                dealign_output_path,
                lines_funcs=[dtool.dealign_a3m, dtool.append_number],
            )

        try:
            cmd = (
                f"mmseqs easy-cluster"
                f" {dealign_output_path}"
                f" {mmseqs2_root / name} {mmseqs2_root / 'tmp'}"
                f" --cluster-reassign"
                f" --min-seq-id {seq_id}"
                f" -c {cov_id}"
                # f" --single-step-clustering"
            )
            execute(cmd, print_off=True)

            raw_lines = dtool.read_lines(input_path, strip=True)
            cluster_repr_path = mmseqs2_root / f"{name}_rep_seq.fasta"
            repr_lines = dtool.read_lines(cluster_repr_path, strip=True)

            repr_a3m_lines = [raw_lines[0], raw_lines[1]]
            for line in repr_lines:
                if dtool.is_comment_line(line):
                    line_num = int(line.strip().split("=")[-1])
                    repr_a3m_lines.extend(
                        [raw_lines[line_num], raw_lines[line_num + 1]]
                    )

            output_path = result_root / f"{name}.fasta"
            dtool.write_lines(output_path, dtool.a3m2fasta(repr_a3m_lines))
            print(
                f"{name} before: {len(raw_lines) // 2} "
                f"after: {len(repr_a3m_lines) // 2}"
            )
        except:
            print("==========")
            print(f"failed: {cmd}")
            traceback.print_exception(*sys.exc_info())
            print("==========")

    if rm_tmp_files:
        shutil.rmtree(dealign_root)
        shutil.rmtree(mmseqs2_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # a3m_dir,strategy_dir,seq_id,cov_id,sample,rm_tmp_files=False
    parser.add_argument("-i", "--input_a3m_path", required=True, type=str)
    parser.add_argument("-o", "--output_a3m_path", required=True, type=str)
    parser.add_argument("--qid", default=0.3)
    parser.add_argument("--cov", default=0.8)
    parser.add_argument("--rm", default=False)

    args = parser.parse_args()

    _run(args.input_a3m_path, args.output_a3m_path, args.qid, args.cov, args.rm)
