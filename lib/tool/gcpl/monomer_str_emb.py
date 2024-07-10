import os
import argparse
import numpy as np
import esm.inverse_folding
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("decoys_dir", type=str, help="decoys dir")
parser.add_argument("out_dir",type=str, help="decoys max msa dir")



args = parser.parse_args()

decoys_dir=args.decoys_dir
out_dir=args.out_dir
name_list = []



alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z']

decoys_dir_list = os.listdir(decoys_dir)

esm_if1 = "/data/protein/datasets_2024/GraphCPLMQA/esm_if1_gvp4_t16_142M_UR50.pt"
model, alphabet = esm.pretrained.load_model_and_alphabet(esm_if1)
model = model.eval()


for decoy in decoys_dir_list:

    # try:
    if decoy.endswith(".pdb"):
        decoy_path = decoys_dir + "/" + decoy
        decoy_if = out_dir+"/"+ decoy.replace(".pdb", ".emb.if.npz")
        print(decoy.replace(".pdb",".emb.if.npz"))

        if os.path.exists(decoy_if):
            print("it is already exits.")
            continue

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
        
        structure = esm.inverse_folding.util.load_structure(decoy_path,chain)
        coords, native_seqs = esm.inverse_folding.util.extract_coords_from_structure(structure)
        rep_list = []
        rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
        
        np.savez_compressed(decoy_if, rep=rep.detach().numpy())
    # except:
    #     with open(decoys_dir+"/err.txt","a") as f:
    #         f.write(decoy+"\n")
