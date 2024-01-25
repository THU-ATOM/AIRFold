import subprocess
from typing import Union
from pathlib import Path
import matplotlib.pyplot as plt

from lib.constant import LDDT_EXECUTE


def compute_lddt(pred_pdb: Union[str, Path], target_pdb: Union[str, Path], CA=True):
    score = None
    if Path(target_pdb).exists() == False:
        raise ValueError(f"target_pdb: {target_pdb} not exist!")
    if Path(pred_pdb).exists() == False:
        raise ValueError(f"pred_pdb: {pred_pdb} not exist!")

    if Path(pred_pdb).exists() and Path(target_pdb).exists():
        report = subprocess.check_output(
            f"{LDDT_EXECUTE} {'-c' if CA else ''} {pred_pdb} {target_pdb}",
            shell=True,
        ).decode("utf-8")
        prefix = "Global LDDT score: "
        for line in report.split("\n"):
            if line.startswith(prefix):
                score = float(line[len(prefix) :])
    return score, report


def get_detailed_lddt(
    pred_pdb: Union[str, Path], target_pdb: Union[str, Path], CA=False
):
    score, report = compute_lddt(pred_pdb=pred_pdb, target_pdb=target_pdb, CA=CA)
    lines = report.strip().split("\n")
    prefix = "Local LDDT Scores:"
    start_idx = None
    for idx, item in enumerate(lines):
        if item.startswith(prefix):
            start_idx = idx + 2
            break
    collects = []
    for line in lines[start_idx:]:
        item = line.strip().split()
        if len(item) == 5:
            offset = 1
        elif len(item) == 6:
            offset = 0
        else:
            raise ValueError("format error")
        if len(item) > 0:
            try:
                collects.append(
                    (
                        int(item[2 - offset]),
                        float(item[4 - offset]) if item[4 - offset] != "-" else -1.0,
                    )
                )
            except:
                print(line)
                raise ValueError("error error")

    return score, tuple(zip(*collects))


def plot_lddts(lddts, Ls=None, dpi=100, fig=True):
    if fig:
        plt.figure(figsize=(8, 5), dpi=100)
    plt.title("lDDT per position")
    for n, plddt in enumerate(lddts):
        plt.plot(plddt, label=f"rank_{n+1}")
    if Ls is not None:
        L_prev = 0
        for L_i in Ls[:-1]:
            L = L_prev + L_i
            L_prev += L_i
            plt.plot([L, L], [0, 100], color="black")
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("LDDT")
    plt.xlabel("Positions")
    return plt
