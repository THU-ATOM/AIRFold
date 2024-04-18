from celery import Celery

from api import run_mmseqs2
import torch
from string import ascii_uppercase, ascii_lowercase
import hashlib, os
from predict import Predictor

WAIT_UNTIL_START = 30
REQUEST_PERIOD = 10

CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "rpc://")
CELERY_BROKER_URL = (
    os.environ.get("CELERY_BROKER_URL", "pyamqp://guest:guest@localhost:5672/"),
)

celery = Celery(
    __name__,
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery.conf.task_routes = {
    "worker.*": {"queue": "queue_rosettafold"},
}

@celery.task(name="rosettafold")
def rosettafoldTask():
    rosetta_main()


def rosetta_main():
    
    import os, time, sys
    if not os.path.isfile("RF2_apr23.pt"):
    # send param download into background
        os.system("(apt-get install aria2; aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/RF2_apr23.pt) &")

    if not os.path.isdir("RoseTTAFold2"):
        print("install RoseTTAFold2")
        os.system("git clone https://github.com/sokrypton/RoseTTAFold2.git")
        os.system("pip -q install py3Dmol")
        os.system("pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html")
        os.system("cd RoseTTAFold2/SE3Transformer; pip -q install --no-cache-dir -r requirements.txt; pip -q install .")
        os.system("wget https://raw.githubusercontent.com/sokrypton/ColabFold/beta/colabfold/mmseqs/api.py")

        # install hhsuite
        print("install hhsuite")
        os.makedirs("hhsuite", exist_ok=True)
        os.system(f"curl -fsSL https://github.com/soedinglab/hh-suite/releases/download/v3.3.0/hhsuite-3.3.0-SSE2-Linux.tar.gz | tar xz -C hhsuite/")


    if os.path.isfile(f"RF2_apr23.pt.aria2"):
        print("downloading RoseTTAFold2 params")
        while os.path.isfile(f"RF2_apr23.pt.aria2"):
            time.sleep(5)

    if not "IMPORTED" in dir():
        if 'RoseTTAFold2/network' not in sys.path:
            os.environ["DGLBACKEND"] = "pytorch"
            sys.path.append('RoseTTAFold2/network')
        if "hhsuite" not in os.environ['PATH']:
            os.environ['PATH'] += ":hhsuite/bin:hhsuite/scripts"

    

    def get_hash(x): return hashlib.sha1(x.encode()).hexdigest()
    alphabet_list = list(ascii_uppercase+ascii_lowercase)
    from collections import OrderedDict, Counter

    IMPORTED = True

    if not "pred" in dir():
    
        print("compile RoseTTAFold2")
        model_params = "RF2_apr23.pt"
    if (torch.cuda.is_available()):
        pred = Predictor(model_params, torch.device("cuda:0"))
    else:
        print ("WARNING: using CPU")
        pred = Predictor(model_params, torch.device("cpu"))

def get_unique_sequences(seq_list):
        unique_seqs = list(OrderedDict.fromkeys(seq_list))
        return unique_seqs


def get_msa(seq, jobname, cov=50, id=90, max_msa=2048,
                mode="unpaired_paired"):

    assert mode in ["unpaired","paired","unpaired_paired"]
    seqs = [seq] if isinstance(seq,str) else seq

    # collapse homooligomeric sequences
    counts = Counter(seqs)
    u_seqs = list(counts.keys())
    u_nums = list(counts.values())

    # expand homooligomeric sequences
    first_seq = "/".join(sum([[x]*n for x,n in zip(u_seqs,u_nums)],[]))
    msa = [first_seq]

    path = os.path.join(jobname,"msa")
    os.makedirs(path, exist_ok=True)
    if mode in ["paired","unpaired_paired"] and len(u_seqs) > 1:
        print("getting paired MSA")
        out_paired = run_mmseqs2(u_seqs, f"{path}/", use_pairing=True)
        headers, sequences = [],[]
        for a3m_lines in out_paired:
            n = -1
            for line in a3m_lines.split("\n"):
                if len(line) > 0:
                    if line.startswith(">"):
                        n += 1
                        if len(headers) < (n + 1):
                            headers.append([])
                            sequences.append([])
                            headers[n].append(line)
            else:
                sequences[n].append(line)
        # filter MSA
        with open(f"{path}/paired_in.a3m","w") as handle:
            for n,sequence in enumerate(sequences):
                handle.write(f">n{n}\n{''.join(sequence)}\n")
        os.system(f"hhfilter -i {path}/paired_in.a3m -id {id} -cov {cov} -o {path}/paired_out.a3m")
        with open(f"{path}/paired_out.a3m","r") as handle:
            for line in handle:
                if line.startswith(">"):
                    n = int(line[2:])
                    xs = sequences[n]
                    # expand homooligomeric sequences
                    xs = ['/'.join([x]*num) for x,num in zip(xs,u_nums)]
                    msa.append('/'.join(xs))

    if len(msa) < max_msa and (mode in ["unpaired","unpaired_paired"] or len(u_seqs) == 1):
        print("getting unpaired MSA")
        out = run_mmseqs2(u_seqs,f"{path}/")
        Ls = [len(seq) for seq in u_seqs]
        sub_idx = []
        sub_msa = []
        sub_msa_num = 0
        for n,a3m_lines in enumerate(out):
            sub_msa.append([])
        with open(f"{path}/in_{n}.a3m","w") as handle:
            handle.write(a3m_lines)
        # filter
        os.system(f"hhfilter -i {path}/in_{n}.a3m -id {id} -cov {cov} -o {path}/out_{n}.a3m")
        with open(f"{path}/out_{n}.a3m","r") as handle:
            for line in handle:
                if not line.startswith(">"):
                    xs = ['-'*l for l in Ls]
                    xs[n] = line.rstrip()
                    # expand homooligomeric sequences
                    xs = ['/'.join([x]*num) for x,num in zip(xs,u_nums)]
                    sub_msa[-1].append('/'.join(xs))
                    sub_msa_num += 1
        sub_idx.append(list(range(len(sub_msa[-1]))))

        while len(msa) < max_msa and sub_msa_num > 0:
            for n in range(len(sub_idx)):
                if len(sub_idx[n]) > 0:
                    msa.append(sub_msa[n][sub_idx[n].pop(0)])
                    sub_msa_num -= 1
                if len(msa) == max_msa:
                    break

    with open(f"{jobname}/msa.a3m","w") as handle:
        for n,sequence in enumerate(msa):
            handle.write(f">n{n}\n{sequence}\n")