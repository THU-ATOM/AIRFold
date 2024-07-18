import os
import torch
import esm
import biotite.structure.io as bsio
from lib.utils.systool import get_available_gpus
import lib.utils.datatool as dtool


def esm_main(seq_name, sequence):
    # get device
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'
    
    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
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


def prediction(sequence, esm_path):
    # get device
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'

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

    esm_pdb_path = os.path.join(esm_path, "model.pdb")
    with open(esm_pdb_path, "w") as f:
        f.write(output)


    # struct = bsio.load_structure(esm_pdb_path, extra_fields=["b_factor"])
    # plddt = struct.b_factor.mean()
    # print("The pLDDT of ESMFold model: %.3f" % plddt)  # this will be the pLDDT
    
    # plddt_json = {"plddt": plddt}
    # esm_json_path = os.path.join(esm_path, "plddt.json")
    # dtool.write_json(esm_json_path, data=plddt_json)
    