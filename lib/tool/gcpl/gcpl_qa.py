import argparse
import os
import shutil
import numpy as np
import torch
from loguru import logger
import lib.tool.gcpl.modules as qa_file
from lib.tool.gcpl.modules import featurize
from lib.tool.gcpl.modules.QA_utils.folding import process_model
from lib.utils.systool import get_available_gpus
from esm import FastaBatchedDataset, pretrained, inverse_folding


def get_seq_embedding(fasta_file, device):
    esm_main="/data/protein/datasets_2024/GraphCPLMQA/esm2_t33_650M_UR50D.pt"
    model, alphabet = pretrained.load_model_and_alphabet(esm_main)
    # model, alphabet = pretrained.esm2_t33_650M_UR50D()
    model.to(device)
    model.eval()

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch=4096, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )

    include_args = ["mean", "per_tok", "bos", "contacts"]
    return_contacts = "contacts" in include_args

    repr_layers = [-1]
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers]

    result = {}
    with torch.no_grad():
        for _, (labels, strs, toks) in enumerate(data_loader):
            
            toks = toks.to(device=device, non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            representations = {
                layer: t.to(device=device) for layer, t in out["representations"].items()
            }

            if return_contacts:
                contacts = out["contacts"].to(device=device)

            for i, _ in enumerate(labels):
                
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in include_args:
                    result["representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].clone()
                        for layer, t in representations.items()
                    }

                if "mean" in include_args:
                    all_layer_sum = 0
                    mean_layer_32 = 0
                    count = 0
                    for layer, t in representations.items():
                        count += 1
                        all_layer_sum += t[i, 1 : len(strs[i]) + 1].clone()
                        if layer ==32:
                            mean_layer_32 = all_layer_sum / 33
                        if layer ==33:
                            last_all = 0.5 * mean_layer_32 + 0.5 * t[i, 1 : len(strs[i]) + 1].clone()
                            result["last_all_rep"] = last_all.cpu().numpy()
                            result["only_last"] = t[i, 1 : len(strs[i]) + 1].clone().cpu().numpy()

                    all_mean=all_layer_sum / 34

                    result["all_mean_rep"] = all_mean.cpu().numpy()

                if "bos" in include_args:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }

                if return_contacts:
                    result["contacts"] = contacts[i, : len(strs[i]), : len(strs[i])].clone()
    return result


def get_str_embedding(decoy_path, device):
    
    # alphabet
    alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    # model setting
    esm_if1 = "/data/protein/datasets_2024/GraphCPLMQA/esm_if1_gvp4_t16_142M_UR50.pt"
    model, alphabet = pretrained.load_model_and_alphabet(esm_if1)
    # model, alphabet = pretrained.esm_if1_gvp4_t16_142M_UR50()
    model.to(device)
    model = model.eval()

    chain_list = []
    chain_temp = "0"
    with open(decoy_path) as f:
        for row in f.readlines():
            if not row.startswith("ATOM"):
                continue
            chain = row[21:22]
            if chain != chain_temp:
                chain_temp = chain
                if "" == chain:
                    continue
                if chain not in alpha:
                    continue
                chain_list.append(chain)

    if len(chain_list) == 0:
        chain=None
    else:
        chain=chain_list[0]
    
    structure = inverse_folding.util.load_structure(decoy_path,chain)
    coords, _ = inverse_folding.util.extract_coords_from_structure(structure)
    rep = inverse_folding.util.get_encoder_output(model, alphabet, coords)
    
    return rep


def evaluation(fasta_file, input_pdbs, tmp_dir, rank_out): 
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'
    
    predicted_result = []
    for decoy_file in input_pdbs:
    
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        
        model_path = "/data/protein/datasets_2024/GraphCPLMQA/QA_Model/GCPL.pkl"
        checkpoint = torch.load(model_path, map_location=device)
        model = qa_file.QA(num_channel=128, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        score = 0.0
        with torch.no_grad():
            
            # get pdb pyroseta feature
            logger.info("step1 --- get pdb pyroseta feature...")
            feature_file = os.path.join(tmp_dir, "features.npz")
            featurize.process(decoy_file, feature_file)
            
            model_coords, _ = process_model(decoy_file)
            (idx, val), (f1d, bert), f2d, _ = qa_file.getData(feature_file, model_coords, bertpath="")
            
            # get sequence embedding
            logger.info("step2 --- get sequence embedding...")
            msa_emb = get_seq_embedding(fasta_file, device)
            node_emb = np.expand_dims(msa_emb["only_last"],0)
            
            # get structure embedding
            logger.info("step3 --- get structure embedding...")
            stru_emb = get_str_embedding(decoy_file, device)
            f1d = np.concatenate([f1d, stru_emb.cpu().numpy()], axis=-1)

            f1d = torch.Tensor(f1d).to(device)
            f2d = torch.Tensor(np.expand_dims(f2d.transpose(2, 0, 1), 0)).to(device)
            idx = torch.Tensor(idx.astype(np.int32)).long().to(device)
            val = torch.Tensor(val).to(device)
            node_emb = torch.Tensor(node_emb).to(device)

            logger.info("step4 --- decoy evaluation...")
            output, _, _ = model(idx, val, f1d, f2d, node_emb, model_coords.to(device))


            lddt = output.p_lddt_pred
            score = np.mean(lddt.cpu().detach().numpy())
        
        shutil.rmtree(tmp_dir)

        predicted_result.append({"predicted_pdb": str(decoy_file), "score": score})
    
    import pickle
    with open(rank_out, 'wb') as f:
        pickle.dump(predicted_result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_fasta", required=True, type=str)
    parser.add_argument("--input_pdbs", required=True, type=str, nargs='*')
    parser.add_argument("--tmp_dir", required=True, type=str)
    parser.add_argument("--rank_out", required=True, type=str)
    
    args = parser.parse_args()
    evaluation(args.input_fasta, args.input_pdbs, args.tmp_dir, args.rank_out)
