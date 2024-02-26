from typing import Any, Dict, Union

from lib.base import BasePathTree
from lib.constant import *

from lib.strategy import StrategyPathTree

# from pipeline.tool.colabfold import AlphaFoldPathTree


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

    @property
    def mmseqs_a3m(self):
        return self.root / "mmseqs_a3m" / f"{self.id}.a3m"

    @property
    def mmseqs_fa(self):
        return self.root / "mmseqs_fa" / f"{self.id}.fasta"

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


class AlphaFoldPathTree(BasePathTree):
    def __init__(self, root: Union[str, Path], request: Dict[str, Any]) -> None:
        super().__init__(root, request)
        self.root = self.root / self.id

    @property
    def time_cost(self):
        return self.root / "time_cost.txt"

    @property
    def template_feat(self):
        return self.root / "template_feat.pkl"

    @property
    def selected_template_feat(self):
        return self.root / "selected_template_feat.pkl"

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
    def alphafold(self):
        return AlphaFoldPathTree(
            self.root / "structure" / self.final_msa_fasta.parent.name,
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