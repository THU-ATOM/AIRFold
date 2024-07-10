from typing import Any, Dict, Union

from lib.base import BasePathTree
from lib.constant import *


def get_pathtree(request: Dict[str, Any]):
    if request["sender"].startswith("cameo"):
        return CAMEOPathTree(CAMEO_DATA_ROOT, request)
    elif request["sender"].startswith("casp15"):
        return CASP15PathTree(CASP15_DATA_ROOT, request)
    elif request["sender"].strip() != "":
        user_path = SENDER_DATA_ROOT / request["sender"]
        user_path.mkdir(parents=True, exist_ok=True)
        return CAMEOPathTree(user_path, request)
    else:
        raise NotImplementedError


class Uniclust30PathTree(BasePathTree):
    def __init__(
        self,
        root: Union[str, Path] = UNICLUST_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        super().__init__(root, request)

    @property
    def data(self):
        return self.root / "UniRef30_2022_02"


class BFDPathTree(BasePathTree):
    def __init__(
        self,
        root: Union[str, Path] = BFD_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        super().__init__(root, request)

    @property
    def data(self):
        return self.root / "bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"


class SmallBFDPathTree(BasePathTree):
    def __init__(
        self,
        root: Union[str, Path] = SBFD_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        super().__init__(root, request)

    @property
    def data(self):
        return self.root / "bfd-first_non_consensus_sequences.fasta"


class AF_Mgnify(BasePathTree):
    def __init__(
        self,
        root: Union[str, Path] = AF_MGNIFY_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        super().__init__(root, request)

    @property
    def data(self):
        return self.root / "mgy_clusters.fa"


class AF_Uniref90(BasePathTree):
    def __init__(
        self,
        root: Union[str, Path] = AF_UNIREF90_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        super().__init__(root, request)

    @property
    def data(self):
        return self.root / "uniref90.fasta"


class SeqPathTree(BasePathTree):
    @property
    def fasta(self):
        return self.root / f"{self.id}.fasta"

    @property
    def aln(self):
        return self.root / f"{self.id}.aln"


class PDBPathTree(BasePathTree):
    @property
    def pdb(self):
        return self.root / "raw_pdb" / f"{self.id}.pdb"


class SearchPathTree(BasePathTree):
    # gather all the paths of the different search method.
    @property
    def integrated_search_a3m(self) -> Path:
        return self.root / "intergrated_a3m" / f"{self.id}.a3m"
    # integrate part
    @property
    def integrated_search_hj_a3m(self) -> Path:
        return self.root / "intergrated_a3m" / f"{self.id}_hj.a3m"
    @property
    def integrated_search_bl_a3m(self) -> Path:
        return self.root / "intergrated_a3m" / f"{self.id}_bl.a3m"
    @property
    def integrated_search_dq_a3m(self) -> Path:
        return self.root / "intergrated_a3m" / f"{self.id}_dq.a3m"
    @property
    def integrated_search_dm_a3m(self) -> Path:
        return self.root / "intergrated_a3m" / f"{self.id}_dm.a3m"
    @property
    def integrated_search_mm_a3m(self) -> Path:
        return self.root / "intergrated_a3m" / f"{self.id}_mm.a3m"
    
    @property
    def integrated_search_a3m_dp(self) -> Path:
        return self.root / "intergrated_a3m_dp" / f"{self.id}.a3m"
    # integrate part duplicated
    @property
    def integrated_search_hj_a3m_dp(self) -> Path:
        return self.root / "intergrated_a3m_dp" / f"{self.id}_hj.a3m"
    @property
    def integrated_search_bl_a3m_dp(self) -> Path:
        return self.root / "intergrated_a3m_dp" / f"{self.id}_bl.a3m"
    @property
    def integrated_search_dq_a3m_dp(self) -> Path:
        return self.root / "intergrated_a3m_dp" / f"{self.id}_dq.a3m"
    @property
    def integrated_search_dm_a3m_dp(self) -> Path:
        return self.root / "intergrated_a3m_dp" / f"{self.id}_dm.a3m"
    @property
    def integrated_search_mm_a3m_dp(self) -> Path:
        return self.root / "intergrated_a3m_dp" / f"{self.id}_mm.a3m"

    @property
    def integrated_search_fa(self) -> Path:
        return self.root / "intergrated_fa" / f"{self.id}.fasta"

    @property
    def hhblist_bfd_uniclust_a3m(self) -> Path:
        return self.root / "hhblist_bfd_ucl_a3m" / f"{self.id}.a3m"

    @property
    def hhblist_bfd_uniclust_aln(self) -> Path:
        return self.root / "hhblist_bfd_ucl_aln" / f"{self.id}.aln"

    @property
    def hhblist_bfd_uniclust_fa(self) -> Path:
        return self.root / "hhblist_bfd_ucl_fa" / f"{self.id}.fasta"

    # mmseqs output path
    @property
    def mmseqs_base(self) -> Path:
        return self.root / "mmseqs" / f"{self.id}"
    
    @property
    def mmseqs_a3m(self):
        return self.root / "mmseqs" / f"{self.id}" / f"{self.id}.a3m"
    @property
    def mmseqs_fa(self):
        return self.root / "mmseqs_fa" / f"{self.id}.fasta"

    # deepmsa output path
    # qmsa
    @property
    def deepqmsa_base(self) -> Path:
        return self.root / "deepqmsa" / f"{self.id}"
    @property
    def deepqmsa_base_tmp(self) -> Path:
        return self.root / "deepqmsa" / f"{self.id}" / "tmp"
    @property
    def deepqmsa_a3m(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.a3m"
    # hhbaln
    @property
    def deepqmsa_hhbaln(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.hhbaln"
    # jacaln
    @property
    def deepqmsa_jacaln(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.jacaln"
    # hh3aln
    @property
    def deepqmsa_hh3aln(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.hh3aln"
    # hh3a3m
    @property
    def deepqmsa_hh3a3m(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.hh3a3m"
    # hhba3m
    @property
    def deepqmsa_hhba3m(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.hhba3m"
    # hmsa3m
    @property
    def deepqmsa_hmsa3m(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.hmsa3m"
    # jaca3m
    @property
    def deepqmsa_jaca3m(self):
        return self.root / "deepqmsa" / f"{self.id}" / f"{self.id}.jaca3m"
    
    # dmsa
    @property
    def deepdmsa_base(self) -> Path:
        return self.root / "deepdmsa" / f"{self.id}"
    @property
    def deepdmsa_base_tmp(self) -> Path:
        return self.root / "deepdmsa" / f"{self.id}" / "tmp"
    @property
    def deepdmsa_a3m(self):
        return self.root / "deepdmsa" / f"{self.id}" / f"{self.id}.a3m"
    # hhbaln
    @property
    def deepdmsa_hhbaln(self):
        return self.root / "deepdmsa" / f"{self.id}" / f"{self.id}.hhbaln"
    # jacaln
    @property
    def deepdmsa_jacaln(self):
        return self.root / "deepdmsa" / f"{self.id}" / f"{self.id}.jacaln"
    # hhba3m
    @property
    def deepdmsa_hhba3m(self):
        return self.root / "deepdmsa" / f"{self.id}" / f"{self.id}.hhba3m"
    # hmsa3m
    @property
    def deepdmsa_hmsa3m(self):
        return self.root / "deepdmsa" / f"{self.id}" / f"{self.id}.hmsa3m"
    # jaca3m
    @property
    def deepdmsa_jaca3m(self):
        return self.root / "deepdmsa" / f"{self.id}" / f"{self.id}.jaca3m"
    
    # mmsa
    @property
    def deepmmsa_base(self):
        return self.root / "deepmmsa" / f"{self.id}"
    @property
    def deepmmsa_base_tmp(self):
        return self.root / "deepmmsa" / f"{self.id}" / "tmp"
    @property
    def deepmmsa_q3jgi(self):
        return self.root / "deepmmsa" / f"{self.id}" / "q3JGI.a3m"
    @property
    def deepmmsa_q4jgi(self):
        return self.root / "deepmmsa" / f"{self.id}" / "q4JGI.a3m"
    @property
    def deepmmsa_djgi(self):
        return self.root / "deepmmsa" / f"{self.id}" / "DeepJGI.a3m"
    
    
    # blast output path
    @property
    def blast_a3m(self):
        return self.root / "blast_a3m" / f"{self.id}.a3m"

    @property
    def blast_fa(self):
        return self.root / "blast_fa" / f"{self.id}.fasta"

    @property
    def blast_whole_fa(self):
        return self.root / "blast_whole" / f"{self.id}_whole.fasta"

    @property
    def jackhammer_uniref90_a3m(self):
        return self.root / "jackhmmer_bfd_a3m" / f"{self.id}.a3m"

    @property
    def jackhammer_uniref90_fa(self):
        return self.root / "jackhmmer_bfd_fa" / f"{self.id}.fasta"

    @property
    def jackhammer_uniref90_sto(self):
        return self.root / "jackhmmer_bfd_sto" / f"{self.id}.sto"

    @property
    def jackhammer_mgnify_a3m(self):
        return self.root / "jackhmmer_mgnify_a3m" / f"{self.id}.a3m"

    @property
    def jackhammer_mgnify_fa(self):
        return self.root / "jackhmmer_mgnify_fa" / f"{self.id}.fasta"

    @property
    def jackhammer_mgnify_sto(self):
        return self.root / "jackhmmer_mgnify_sto" / f"{self.id}.sto"

    @property
    def in_fasta(self):
        return self.root / "in_fasta" / f"{self.id}.fasta"

    @property
    def template_hits(self):
        return self.root / "template_hits" / f"{self.id}.hits.pkl"

    @property
    def template_feat(self):
        return self.root / "template_feat" / f"{self.id}.pkl"
    
    @property
    def selected_template_feat(self):
        return self.root / "selected_template_feat" / f"{self.id}.pkl"

class StrategyPathTree(BasePathTree):
    @property
    def strategy_list(self):
        # return a list of file directory, each stands for an output of the intermiediate results.
        def parse_strgy(s_: Dict[str, Any]):
            # this function parse a strategy from a dict of
            re = ""
            for para_ in s_.keys():
                re = re + "_" + para_[:2] + "_{}".format(s_[para_])
            return re

        if "idle" in self.request["run_config"]["msa_select"].keys():
            # deal with idle cases
            return []

        # deal with the case of manual selection
        if "manual" in self.request["run_config"]["msa_select"].keys():
            # to make it compatible, the manual case should only have a empty params
            return [self.root / "manual" / f"{self.id}.a3m"]

        str_dict = self.request["run_config"]["msa_select"]
        path_l = []
        p_ = self.root
        # print(p_)
        for method_ in str_dict.keys():
            # print(p_)
            p_ = p_ / (method_[:5] + parse_strgy(str_dict[method_]["least_seqs"]))
            # path_l.append(p_ / f"{self.id}.a3m")
            path_l.append(p_ / f"{self.id}")
        # return the list of
        return path_l

    @property
    def final_fasta(self):
        if len(self.strategy_list) > 0:
            return self.strategy_list[-1]
        else:
            return None    

class AlphaFoldPathTree(BasePathTree):
    def __init__(self, root: Union[str, Path], request: Dict[str, Any]) -> None:
        super().__init__(root, request)
        self.root = self.root / self.id / "alpha"

    @property
    def time_cost(self):
        return self.root / "time_cost.txt"
    
    @property
    def processed_feat(self):
        return self.root / "processed_feat"

    @property
    def relaxed_pdbs(self):
        return list(sorted(self.root.glob("rank_*_relaxed.pdb")))

    @property
    def submit_pdbs(self):
        return list(sorted(self.root.glob("model_*_relaxed.pdb")))

    @property
    def unrelaxed_pdbs(self):
        return list(sorted(self.root.glob("*_unrelaxed.pdb")))

    @property
    def result(self):
        return self.root / "result.json"
    
    @property
    def plddt_results(self):
        return self.root / "plddt_results.json"

    @property
    def lddt(self):
        return self.root / "lddt" / "lddt.json"

    @property
    def msa_pickle(self):
        return self.root / "msa.pickle"

    @property
    def input_a3m(self):
        return self.root / "input_msa.a3m"

    @property
    def msa_filtered_pickle(self):
        return self.root / "msa_filtered.pickle"

    @property
    def log(self):
        return self.root / "log.txt"

    @property
    def plddt_image(self):
        return self.root / "predicted_LDDT.png"

    @property
    def msa_coverage_image(self):
        return self.root / "msa_coverage.png"

    @property
    def model_files(self):
        files = []
        for item in self.relaxed_pdbs:
            key = "_".join(item.name.split("_")[:-1])
            itemfiles = {}
            model_key = "_".join(item.name.split("_")[2:4])
            itemfiles["relaxed_pdb"] = item
            itemfiles["unrelaxed_pdb"] = self.root / f"{key}_unrelaxed.pdb"
            itemfiles["plddt"] = self.root / model_key / "result.json"
            itemfiles["image"] = self.root / f"{key}.png"
            itemfiles["conformation"] = self.root / f"{item.stem}_conformation.png"
            files.append(itemfiles)
        if len(files) > 1:
            files = sorted(files, key=lambda x: x["relaxed_pdb"])
        return files


class RoseTTAFoldPathTree(BasePathTree):
    def __init__(self, root: Union[str, Path], request: Dict[str, Any]) -> None:
        super().__init__(root, request)
        self.root = self.root / self.id / "rose"

    @property
    def time_cost(self):
        return self.root / "time_cost.txt"

    @property
    def relaxed_pdbs(self):
        return list(sorted(self.root.glob("rank_*_relaxed.pdb")))

    @property
    def submit_pdbs(self):
        return list(sorted(self.root.glob("model_*_relaxed.pdb")))

    @property
    def unrelaxed_pdbs(self):
        return list(sorted(self.root.glob("*_unrelaxed.pdb")))

    @property
    def result(self):
        return self.root / "result.json"

    @property
    def lddt(self):
        return self.root / "lddt" / "lddt.json"

    @property
    def msa_pickle(self):
        return self.root / "msa.pickle"

    @property
    def input_a3m(self):
        return self.root / "input_msa.a3m"

    @property
    def msa_filtered_pickle(self):
        return self.root / "msa_filtered.pickle"

    @property
    def log(self):
        return self.root / "log.txt"

    @property
    def plddt_image(self):
        return self.root / "predicted_LDDT.png"

    @property
    def msa_coverage_image(self):
        return self.root / "msa_coverage.png"

    @property
    def model_files(self):
        files = []
        for item in self.relaxed_pdbs:
            key = "_".join(item.name.split("_")[:-1])
            itemfiles = {}
            model_key = "_".join(item.name.split("_")[2:4])
            itemfiles["relaxed_pdb"] = item
            itemfiles["unrelaxed_pdb"] = self.root / f"{key}_unrelaxed.pdb"
            itemfiles["plddt"] = self.root / model_key / "result.json"
            itemfiles["image"] = self.root / f"{key}.png"
            itemfiles["conformation"] = self.root / f"{item.stem}_conformation.png"
            files.append(itemfiles)
        if len(files) > 1:
            files = sorted(files, key=lambda x: x["relaxed_pdb"])
        return files


class MQEPathTree(BasePathTree):
    def __init__(self, root: Union[str, Path], request: Dict[str, Any]) -> None:
        super().__init__(root, request)
        self.root = self.root / self.id

    @property
    def enqa(self)-> Path:
        return self.root / "enqa"
    
    @property
    def enqa_temp(self)-> Path:
        return self.root / "enqa" / "temp"
    
    @property
    def enqa_rankfile(self)-> Path:
        return self.root / "enqa" / "rank.json"
    
    @property
    def gcpl(self)-> Path:
        return self.root / "gcpl"
    
    @property
    def gcpl_temp(self)-> Path:
        return self.root / "gcpl" / "temp"
    
    @property
    def gcpl_rankfile(self)-> Path:
        return self.root / "gcpl" / "rank.json"



class CAMEOPathTree(BasePathTree):
    def __init__(
        self,
        root: Union[str, Path] = CAMEO_DATA_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        root = Path(root) / request["name"].split("_")[0]
        super().__init__(root, request)

    @property
    def seq(self) -> SeqPathTree:
        return SeqPathTree(self.root / "seq", self.request)

    @property
    def search(self) -> SearchPathTree:
        return SearchPathTree(self.root / "search", self.request)

    @property
    def strategy(self) -> StrategyPathTree:
        return StrategyPathTree(self.root / "strategy", self.request)

    @property
    def final_msa_fasta(self) -> Path:
        # if self.request["segment"] !=
        return (
            self.strategy.final_fasta
            if self.strategy.final_fasta is not None
            else self.search.integrated_search_fa
        )

    @property
    def struc_root(self):
        return self.root / "structure"
        
    @property
    def alphafold(self):
        return AlphaFoldPathTree(
            self.root / "structure" / self.final_msa_fasta.parent.name,
            self.request,
        )
    
    @property
    def rosettafold2(self):
        return RoseTTAFoldPathTree(
            self.root / "structure" / self.final_msa_fasta.parent.name,
            self.request,
        )
    
    @property
    def mqe(self):
        return MQEPathTree(
            self.root / "mqe",
            self.request,
        )

    @property
    def pdb(self) -> PDBPathTree:
        return PDBPathTree(self.root / "pdb", self.request)

    @property
    def uniclust30(self) -> Uniclust30PathTree:
        return Uniclust30PathTree(request=self.request)

    @property
    def bfd(self) -> BFDPathTree:
        return BFDPathTree(request=self.request)

    @property
    def afuniref(self) -> AF_Uniref90:
        return AF_Uniref90(request=self.request)

    @property
    def afmgnify(self) -> AF_Mgnify:
        return AF_Mgnify(request=self.request)


class CASP15PathTree(CAMEOPathTree):
    def __init__(
        self,
        root: Union[str, Path] = CASP15_DATA_ROOT,
        request: Dict[str, Any] = None,
    ) -> None:
        super().__init__(root, request)
