# from string import ascii_uppercase, ascii_lowercase
import hashlib, re, os
import numpy as np
import torch
from jax.tree_util import tree_map
from scipy.special import softmax
import gc
import os
from celery import Celery
import torch

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
    "worker.*": {"queue": "queue_esmfold"},
}


@celery.task(name="esmfold")
def esmfoldTask():
    esm_main()


def parse_output(output):
    pae = (output["aligned_confidence_probs"][0] * np.arange(64)).mean(-1) * 31
    plddt = output["plddt"][0,:,1]

    bins = np.append(0,np.linspace(2.3125,21.6875,63))
    sm_contacts = softmax(output["distogram_logits"],-1)[0]
    sm_contacts = sm_contacts[...,bins<8].sum(-1)
    xyz = output["positions"][-1,0,:,1]
    mask = output["atom37_atom_exists"][0,:,1] == 1
    o = {"pae":pae[mask,:][:,mask],
        "plddt":plddt[mask],
        "sm_contacts":sm_contacts[mask,:][:,mask],
        "xyz":xyz[mask]}
    return o

def get_hash(x): 
    
    return hashlib.sha1(x.encode()).hexdigest()


def esm_main():
    version = "1" # @param ["0", "1"]
    model_name = "esmfold_v0.model" if version == "0" else "esmfold.model"
    import os, time
    if not os.path.isfile(model_name):
        # download esmfold params
        os.system("apt-get install aria2 -qq")
        os.system(f"aria2c -q -x 16 https://colabfold.steineggerlab.workers.dev/esm/{model_name} &")

    if not os.path.isfile("finished_install"):
        # install libs
        print("installing libs...")
        os.system("pip install -q omegaconf pytorch_lightning biopython ml_collections einops py3Dmol modelcif")
        os.system("pip install -q git+https://github.com/NVIDIA/dllogger.git")

        print("installing openfold...")
        # install openfold
        os.system(f"pip install -q git+https://github.com/sokrypton/openfold.git")

        print("installing esmfold...")
        # install esmfold
        os.system(f"pip install -q git+https://github.com/sokrypton/esm.git")
        os.system("touch finished_install")

    # wait for Params to finish downloading...
    while not os.path.isfile(model_name):
        time.sleep(5)
    if os.path.isfile(f"{model_name}.aria2"):
        print("downloading params...")
    while os.path.isfile(f"{model_name}.aria2"):
        time.sleep(5)


    # alphabet_list = list(ascii_uppercase+ascii_lowercase)

    jobname = "test" #@param {type:"string"}
    jobname = re.sub(r'\W+', '', jobname)[:50]

    sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE" #@param {type:"string"}
    sequence = re.sub("[^A-Z:]", "", sequence.replace("/",":").upper())
    sequence = re.sub(":+",":",sequence)
    sequence = re.sub("^[:]+","",sequence)
    sequence = re.sub("[:]+$","",sequence)
    copies = 1 #@param {type:"integer"}
    if copies == "" or copies <= 0: copies = 1
    sequence = ":".join([sequence] * copies)
    num_recycles = 3 #@param ["0", "1", "2", "3", "6", "12", "24"] {type:"raw"}
    chain_linker = 25

    ID = jobname+"_"+get_hash(sequence)[:5]
    seqs = sequence.split(":")
    lengths = [len(s) for s in seqs]
    length = sum(lengths)
    print("length",length)

    u_seqs = list(set(seqs))
    if len(seqs) == 1: mode = "mono"
    elif len(u_seqs) == 1: mode = "homo"
    else: mode = "hetero"

    if "model" not in dir() or model_name != model_name_:
        if "model" in dir():
            # delete old model from memory
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    model = torch.load(model_name)
    model.eval().cuda().requires_grad_(False)
    model_name_ = model_name

    # optimized for Tesla T4
    if length > 700:
        model.set_chunk_size(64)
    else:
        model.set_chunk_size(128)

    torch.cuda.empty_cache()
    output = model.infer(sequence,
                        num_recycles=num_recycles,
                        chain_linker="X"*chain_linker,
                        residue_index_offset=512)

    pdb_str = model.output_to_pdb(output)[0]
    output = tree_map(lambda x: x.cpu().numpy(), output)
    ptm = output["ptm"][0]
    plddt = output["plddt"][0,...,1].mean()
    O = parse_output(output)
    print(f'ptm: {ptm:.3f} plddt: {plddt:.3f}')
    os.system(f"mkdir -p {ID}")
    prefix = f"{ID}/ptm{ptm:.3f}_r{num_recycles}_default"
    np.savetxt(f"{prefix}.pae.txt",O["pae"],"%.3f")
    with open(f"{prefix}.pdb","w") as out:
        out.write(pdb_str)