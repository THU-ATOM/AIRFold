import torch
from lib.strategy.plmsim.plmsearch_util.util import cos_similarity, tensor_to_list

def main(query_embedding_dic, target_embedding_dic, device, model, least_seqs):
    with torch.no_grad():
        query_proteins = list(query_embedding_dic.keys())
        query_embedding = torch.stack([query_embedding_dic[key] for key in query_proteins])
        query_embedding = query_embedding.to(device)

        target_proteins = list(target_embedding_dic.keys())
        target_embedding = torch.stack([target_embedding_dic[key] for key in target_proteins])
        target_embedding = target_embedding.to(device)

        similarity_dict = {}
        for protein in query_proteins:
            similarity_dict[protein] = {}

        cos_matrix = cos_similarity(query_embedding, target_embedding)
        cos_matrix_list = tensor_to_list(cos_matrix)
        
        sim_matrix = model.predict(query_embedding, target_embedding)
        sim_matrix_list = tensor_to_list(sim_matrix)

        for i, query_protein in enumerate(query_proteins):
            for j, target_protein in enumerate(target_proteins):
                similarity_dict[query_protein][target_protein] = cos_matrix_list[i][j] if (cos_matrix_list[i][j]>0.995) else cos_matrix_list[i][j] * sim_matrix_list[i][j]

        protein_pair_dict = {}
        for protein in query_proteins:
            protein_pair_dict[protein] = []


        for query_protein in query_proteins:
            for target_protein in similarity_dict[query_protein]:
                protein_pair_dict[query_protein].append((target_protein, similarity_dict[query_protein][target_protein]))


        for query_protein in query_proteins:
            sorted_pair = sorted(protein_pair_dict[query_protein], key=lambda x:x[1], reverse=True)
            if len(sorted_pair) <= least_seqs:
                protein_pair_dict[query_protein] = sorted_pair
            else:
                protein_pair_dict[query_protein] = sorted_pair[:least_seqs]

    return protein_pair_dict
