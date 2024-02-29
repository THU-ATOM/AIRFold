#this file implement the clustering strategy for msa with ccmpred

import sys
sys.path.append("/code/Projects/psp-pipeline")

import argparse
from typing import Sequence
import torch
from lib.utils.execute import execute
import lib.utils.datatool as dtool
from pathlib import Path
import numpy as np
# import os
# import tensorflow as tf
# from sklearn.cluster import AffinityPropagation
# from pipeline.tool.colabfold.alphafold.common.residue_constants import HHBLITS_AA_TO_ID,shape_helpers
# import distance
import torch.nn.functional as F
#
HHBLITS_AA_TO_ID = {
    'A': 0,
    'B': 2,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'J': 20,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'O': 20,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'U': 1,
    'V': 17,
    'W': 18,
    'X': 20,
    'Y': 19,
    'Z': 3,
    '-': 21,
}

class ClusterStrategy(object):
    def __init__(self,original_msa_file,cluster_folder):
        self.original_msa = original_msa_file
        self.clustered = cluster_folder
    def gen_cluster_mask(self):
        raise NotImplementedError
    def cluster_(self):
        raise NotImplementedError


def make_msa_features(
    msas: Sequence[Sequence[str]]):
  """Constructs a feature dict of MSA features."""
  if not msas:
    raise ValueError('At least one MSA must be provided.')
  int_msa = []
  seen_sequences = set()
  
  for sequence_index, sequence in enumerate(msas):
    if sequence in seen_sequences:
        continue
    seen_sequences.add(sequence)
    int_msa.append(
        [HHBLITS_AA_TO_ID[res] for res in sequence])
#   print(int_msa)
  msa = np.array(int_msa, dtype=np.int32)
  return msa




def nearest_neighbor_clusters(msa, cluster_center, gap_agreement_weight=0.):
  """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

  # Determine how much weight we assign to each agreement.  In theory, we could
  # use a full blosum matrix here, but right now let's just down-weight gap
  # agreement because it could be spurious
  weights = [1] * 21 + [gap_agreement_weight] + [0]
  weights = torch.from_numpy(np.asarray(weights,dtype=np.float32))

  sample_msa = F.one_hot(msa,23)
  cluster_center = F.one_hot(cluster_center,23)

  
  num_seq, num_res, _ = sample_msa.size()
  sample_msa = sample_msa.view(num_seq, num_res * 23)

  center_num, _, _ = cluster_center.size()

  cluster_center = (torch.mul(cluster_center,weights)).view(center_num, num_res * 23)

  agreement = torch.matmul(sample_msa.to(torch.float32).cuda(), cluster_center.cuda().t())

  assignment = torch.argmax(agreement, 1).cpu()

  return assignment


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


def update_cluster_profile(msa,assignment,c_num = 5, gap_agreement_weight=0.):
    sample_msa = F.one_hot(msa,23).to(torch.float32)
    num_seq, num_res, _ = sample_msa.size()


    # c_num, _= cluster_center.size()
    profile = unsorted_segment_sum(sample_msa, assignment, c_num)
    profile = F.log_softmax(profile, dim=-1)

    sample_msa = sample_msa.view(num_seq, num_res * 23)

    # import pdb
    # pdb.set_trace()
    weights = [1] * 21 + [gap_agreement_weight] + [0]
    weights = torch.from_numpy(np.asarray(weights,dtype=np.float32))

    profile = (torch.mul(profile,weights)).view(c_num, num_res * 23)

    likelihood = torch.matmul(sample_msa.cuda(), profile.cuda().t())
    assignment = torch.argmax(likelihood, 1).cpu()

    return assignment, profile


class CcmpredCluster(ClusterStrategy):
    def __init__(self, original_msa_file:str, cluster_file:str, column:int , clusters:int, threthold=0.3,metric="likelihood"):
        super().__init__(original_msa_file, cluster_file)
        self.column = column
        self.metric = metric
        self.threthold = threthold
        self.cluster_nums = clusters

    def gen_cluster_mask(self):
        #return a mask which use to get the mask which 
        name = self.original_msa.split('/')[-1].split('.')[0]
        file_name = Path("/tmp") / Path(name+".mat")
        aln_file = Path("/tmp") / Path(name+".aln")
        dtool.fasta2aln(self.original_msa,aln_file)
        try:
            cmd = f"ccmpred {aln_file} {file_name}"
        except:
            print("ccmpred error")
        execute(cmd)
        matrix = np.loadtxt(file_name)
        lower_tri = np.tril(matrix)
        thr_num = int((lower_tri>0).sum() * self.threthold)
        n_largest= np.sort(lower_tri, axis=None)[-thr_num] # n_largest 
        mask_ = matrix[self.column,:] > (n_largest - 1e-10)

        print(f"used index is {mask_.sum()} and the total length is {len(mask_)}")

        return mask_
    
    def cluster_torch(self,epochs = 2, gap_agreement_weight=0.):
        mask = self.gen_cluster_mask()
        cluster_nums = self.cluster_nums
        lines = dtool.fasta2list(self.original_msa)
        original_seq = lines[0]
        lines = lines[1:]
        msa =  make_msa_features(lines)
        msa = torch.from_numpy(msa).to(torch.int64)


        mask = torch.from_numpy(mask)

        msa[:,mask] = torch.tensor(21)

        #initial cluster center:
        perm = torch.randperm(msa.size(0))
        idx = perm[:cluster_nums]
        cluster_center = msa[idx,:]
        cluster_alignment = nearest_neighbor_clusters(msa, cluster_center)

        for i in range(epochs):
            cluster_alignment, cluster_profile = update_cluster_profile(msa,cluster_alignment,cluster_nums, gap_agreement_weight)
            print(f"epoch {i} finished")
        
        cluster_alignment = cluster_alignment.cpu().numpy()

        results = {}

        #init result collector
        for i in range(cluster_nums):
            results[i] = [original_seq]
        
        for i in range(len(cluster_alignment)):
            results[cluster_alignment[i]].append(lines[i])
        
        Path(self.clustered).mkdir(exist_ok=True)
        name = self.original_msa.split('/')[-1].split('.')[0]
        cluster_center = cluster_center.cpu().numpy()

        with open(self.clustered+"/"+name+"_cluster.npz","wb") as f:
            np.save(f,cluster_profile)

        for i in results.keys():
            dtool.list2fasta(Path(self.clustered) / name /("cluster_"+str(i)+".fasta"),results[i])
                        # dtool.list2fasta(Path(self.clustered) / Path(name+"_cluster_"+str(i)+".fasta"),results[i])


    # def cluster_Aff(self,dam=0.5):
    #     mask = self.gen_cluster_mask()
    #     with open(self.original_msa) as f:
    #         lines = f.readlines()
    #         lines = [line.strip() for line in lines]
    #     seq = lines[0]
    #     lines = np.asarray(lines[1:])
    #     if self.metric == "leven":
    #         dist = distance.levenshtein
    #     else:
    #         raise NotImplementedError
        
    #     similarity = -1*np.array([[dist(w1 * mask,w2* mask) for w1 in lines] for w2 in lines])

    #     af = AffinityPropagation(affinity="precomputed", damping=dam)

    #     af.fit(similarity)

    #     cluster_result = []

    #     for cluster_id in np.unique(af.labels_):
    #         exemplar = lines[af.cluster_centers_indices_[cluster_id]]
    #         cluster = seq + np.unique(lines[np.nonzero(af.labels_==cluster_id)]).to_list()
    #         cluster_result.append({"exemplar":exemplar,"cluster":cluster})
        
    #     os.mkdir(self.clustered,exists_ok=True)
    #     name = self.original_msa.split('/')[-1].split('.')[0]
    #     with open(self.clustered+"/"+name+".cluster","w") as f:
    #         for cluster in cluster_result:
    #             f.write(f"{cluster['exemplar']}\n")
    #             #f.write(f"{cluster['cluster']}\n")
    #     for i,cluster in enumerate(cluster_result):
    #         dtool.list2fasta((Path(self.clustered) / name /"cluster_"+str(i)+".fasta"),cluster['cluster'])
        
    #     print(f"cluster result is saved in {self.clustered}")

# c = CcmpredCluster("/data/protein/CASP15/data/2022-05-06/search/intergrated_fa/2022-05-06_T1110___Q9zm@s@50@e@100.fasta","/data/train_log/songyuxuan/tmp",160)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cluster the msa file")
    parser.add_argument("-i","--input",type=str,default = "/data/protein/CASP15/data/2022-05-06/search/intergrated_fa/2022-05-06_T1109.fasta",help="the input msa file")
    parser.add_argument("-o","--output",type=str,default= "/data/train_log/songyuxuan/tmp",help="the output folder")
    parser.add_argument("-c","--column",type=int, default= 182, help="the column to cluster")
    parser.add_argument("-n","--cluster_nums",type=int,default=3,help="the number of cluster")
    parser.add_argument("-t","--threthold",type=float,default = 0.5, help="the threthold to cluster")
    parser.add_argument("-m","--metric",type=str,default="likelihood",help="the metric to cluster")
    args = parser.parse_args()
    cluster = CcmpredCluster(args.input,args.output,args.column,args.cluster_nums,args.threthold,args.metric)
    cluster.cluster_torch()