# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import pickle

from esm import FastaBatchedDataset, pretrained

def main(esm_model_path, fasta, embedding_result):
    esm_model, alphabet = pretrained.load_model_and_alphabet(esm_model_path)
    esm_model.eval()

    if torch.cuda.is_available():
        esm_model = esm_model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(16384, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    print(f"Read {fasta} with {len(dataset)} sequences")

    embedding_result_dic = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
               toks = toks.to(device="cuda", non_blocking=True)

            out = esm_model(toks, repr_layers=[33], return_contacts=False)["representations"][33]

            for i, label in enumerate(labels):
                #get mean embedding
                esm_embedding = out[i, 1 : len(strs[i]) + 1].mean(0).clone().cpu()
                embedding_result_dic[label] = esm_embedding
        
        with open(embedding_result, 'wb') as handle:
            pickle.dump(embedding_result_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)