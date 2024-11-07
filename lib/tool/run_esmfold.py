import argparse
import os
import torch
import esm
import numpy as np
from lib.utils.systool import get_available_gpus
import lib.utils.datatool as dtool
import random
from Bio import SeqIO
from  loguru import logger


def esm_main(seq_name, sequence):
    # get device
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'
    # Load ESM-2 model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_main="/data/protein/datasets_2024/GraphCPLMQA/esm2_t33_650M_UR50D.pt"
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_main)
    batch_converter = alphabet.get_batch_converter()
    model.to(device)
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    seq_data = [(seq_name, sequence)]
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein3",  "K A <mask> I S Q"),
    # ]
    batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    print(sequence_representations)
    # Look at the unsupervised self-attention map contact predictions
    # import matplotlib.pyplot as plt
    # for (_, seq), tokens_len, attention_contacts in zip(seq_data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    #     plt.title(seq)
    #     plt.show()


def read_fasta(fn_fasta):
    prot2seq = {}
    with open(fn_fasta) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq)
            prot = record.id
            prot2seq[prot] = seq
    return list(prot2seq.keys()), prot2seq


def prediction(sequence, esm_pdb_path, random_seed):
    # get device
    random.seed(random_seed)
    np.random.seed(random_seed)
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'
    logger.info(f"The device for running esmfold predicion: {device}")
    
    logger.info("Loading model esmfold_v1...............")
    model = esm.pretrained.esmfold_v1()
    
    model = model.eval()
    model.to(device)

    # Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
    # Lower sizes will have lower memory requirements at the cost of increased speed.
    # model.set_chunk_size(128)

    # sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    # Multimer prediction can be done with chains separated by ':'

    with torch.no_grad():
        output = model.infer_pdb(sequence)

    dtool.write_text_file(plaintext=output, path=esm_pdb_path)

    # struct = bsio.load_structure(esm_pdb_path, extra_fields=["b_factor"])
    # plddt = struct.b_factor.mean()
    # print("The pLDDT of ESMFold model: %.3f" % plddt)  # this will be the pLDDT
    # plddt_json = {"plddt": plddt}
    # esm_json_path = os.path.join(esm_path, "plddt.json")
    # dtool.write_json(esm_json_path, data=plddt_json)


if __name__ == "__main__":
    logger.info("RUNNING ESMFOLD PREDICTION START!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, required=True)
    parser.add_argument("--pdb_root", type=str, required=True)
    parser.add_argument("--random_seed", type=int, required=True)
    parser.add_argument("--model_names", type=str, required=True, nargs='*')
    
    args = parser.parse_args()
    
    # get sequence
    seq_names, prot2seq = read_fasta(args.fasta_file)
    logger.info(f"++++ RUNNING ESMFOLD PREDICTION FOR: {seq_names[0]} ++++")
    sequence = list(prot2seq.values())[0]
    
    for idx, model_name in enumerate(args.model_names):
        logger.info(f"----- {model_name} RUNNING -----")
        pdb_path = str(os.path.join(args.pdb_root, model_name)) + "_relaxed.pdb"
        prediction(sequence, pdb_path, args.random_seed+idx)
    logger.info("RUNNING ESMFOLD PREDICTION FINISHED!")