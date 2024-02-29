import argparse
import json
import os
import platform
from pathlib import Path

import tensorflow as tf
from Bio import SeqIO
import numpy as np
import tqdm
import pickle

from pipeline.tool.colabfold import colabfold as cf
from pipeline.tool.colabfold import colabfold_alphafold as cf_af
from pipeline.constant import HHFILTER_PATH


tf.config.set_visible_devices([], "GPU")


os.system("export TF_FORCE_UNIFIED_MEMORY=1 ")
os.system("export NVIDIA_VISIBLE_DEVICES='all' ")
os.system("export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0' ")


def readfastafile(fastafile):
    records = list(SeqIO.parse(fastafile, "fasta"))
    if len(records) != 1:
        raise ValueError("Input FASTA file must have a single ID/sequence.")
    else:
        return records[0].id, records[0].seq


def save_results(output_dir, result_prefix, result_data):
    Path(os.path.join(output_dir, result_prefix)).mkdir(
        parents=True, exist_ok=True
    )
    json_output_path = os.path.join(output_dir, result_prefix, f"result.json")
    with open(json_output_path, "w") as f:
        f.write(result_data)
    print(result_prefix, result_data)
    return


def do_save_to_txt(filename, adj, dists, sequence):
    adj = np.asarray(adj)
    dists = np.asarray(dists)
    L = len(adj)
    with open(filename, "w") as out:
        out.write("i\tj\taa_i\taa_j\tp(cbcb<8)\tmaxdistbin\n")
        for i in range(L):
            for j in range(i + 1, L):
                if dists[i][j] < 21.68 or adj[i][j] >= 0.001:
                    line = f"{i}\t{j}\t{sequence[i]}\t{sequence[j]}\t{adj[i][j]:.3f}"
                    line += (
                        f"\t>{dists[i][j]:.2f}"
                        if dists[i][j] == 21.6875
                        else f"\t{dists[i][j]:.2f}"
                    )
                    out.write(f"{line}\n")


def alphafold_predict(
    seq_name: str,
    input_fasta: Path,
    msa_path: Path,
    output_dir: Path,
    seqcov: float = 0,
    seqqid: float = 0,
    homooligomer: str = "1",
    msa_method: str = "single_sequence",
    add_msa: bool = True,
    precomputed: str = None,
    params_loc: Path = None,
    pair_mode: str = "unpaired",
    pair_cov: int = 50,
    pair_qid: int = 20,
    rank_by: str = "pLDDT",
    use_turbo: bool = False,
    max_msa: str = "512:1024",
    num_models: int = 5,
    use_ptm: bool = False,
    num_ensemble: int = 1,
    max_recycles: int = 3,
    tol: float = 0,
    is_training: bool = False,
    num_samples: int = 1,
    num_relax: str = "Top5",
):
    """_summary_

    Parameters
    ----------
    seq_name : str
        The symbol of sequence
    input_fasta : Path
        Path to a FASTA file. Required.
    msa_path : Path
        Path to a custom msa FASTA file. Required.
    output_dir : Path
        Path to a directory that will store the results.
    seqcov : float, optional
        The coverage filter for msa, by default 0
    seqqid : float, optional
        The indentity filter for msa, by default 0
    homooligomer : str, optional
        Define number of copies in a homo-oligomeric assembly. For example,
        sequence:ABC:DEF, homooligomer: 2:1, the first protein ABC will be
        modeled as a homodimer (2 copies) and second DEF a monomer (1 copy).
        Default is 1.
    msa_method : str, optional
        Options to generate MSA.
        mmseqs2 - FAST method from ColabFold (default)
        single_sequence - use single sequence input.
        precomputed - specify 'msa.pickle' file generated previously if you have.
        Default is 'single_sequence'."
    add_msa : bool, optional
        whether we need to add msa/use custom msa, by default True
    precomputed : str, optional
        Specify the file path of a precomputed pickled msa from previous run,
        by default None
    params_loc : Path, optional
        The root path of AlphaFold parameters, by default None
    pair_mode : str, optional
        Experimental option for protein complexes. Pairing currently only
        supported for proteins in same operon (prokaryotic genomes).
        unpaired - generate separate MSA for each protein. (default)
        unpaired+paired - attempt to pair sequences from the same operon within the genome.
        paired - only use sequences that were successfully paired.
        Default is 'unpaired'."
    pair_cov : int, optional
        Options to prefilter each MSA before pairing. It might help if there
        are any paralogs in the complex. prefilter each MSA to minimum coverage
        with query (%%) before pairing.
        Default is 50.
    pair_qid : int, optional
        Options to prefilter each MSA before pairing. It might help if there
        are any paralogs in the complex. prefilter each MSA to minimum sequence
        identity with query (%%) before pairing. Default is 20.
    rank_by : str, optional
        specify metric to use for ranking models (For protein-protein complexes,
        we recommend pTMscore). Default is 'pLDDT'.
    use_turbo : bool, optional
        introduces a few modifications (compile once, swap params, adjust
        max_msa) to speedup and reduce memory requirements.
        Disable for default behavior.
    max_msa : _type_, optional
        max_msa defines: max_msa_clusters:max_extra_msa number of sequences to
        use. This option ignored if use_turbo is disabled. Default is '512:1024'.
    num_models : int, optional
        Specify how many model params to try. (Default is 5)
    use_ptm : bool, optional
        uses Deepmind's ptm finetuned model parameters to get PAE per structure.
        Disable to use the original model params.
        (Disabling may give alternative structures.)
    num_ensemble : int, optional
        the trunk of the network is run multiple times with different random
        choices for the MSA cluster centers. (1=default, 8=casp14 setting)
    max_recycles : int, optional
        controls the maximum number of times the structure is fed back into the
        neural network for refinement. (default is 3)
    tol : float, optional
        tolerance for deciding when to stop (CA-RMS between recycles)
    is_training : bool, optional
        enables the stochastic part of the model (dropout), when coupled with
        num_samples can be used to 'sample' a diverse set of structures.
        False (NOT specifying this option) is recommended at first.
    num_samples : int, optional
        number of random_seeds to try. Default is 1.
    num_relax : str, optional
        num_relax is 'None' (default), 'Top1', 'Top5' or 'All'. Specify how
        many of the top ranked structures to relax.
    """
    args = argparse.Namespace(
        params_loc=params_loc,
        input=input_fasta,
        c_msa=msa_path,
        add_msa=add_msa,
        output_dir=output_dir,
        seqcov=seqcov,
        seqqid=seqqid,
        seqn=seq_name,
        homooligomer=homooligomer,
        msa_method=msa_method,
        precomputed=precomputed,
        pair_mode=pair_mode,
        pair_cov=pair_cov,
        pair_qid=pair_qid,
        rank_by=rank_by,
        max_msa=max_msa,
        num_models=num_models,
        use_ptm=use_ptm,
        num_ensemble=num_ensemble,
        max_recycles=max_recycles,
        tol=tol,
        is_training=is_training,
        num_samples=num_samples,
        num_relax=num_relax,
    )
    alphafold_predict_run(args)


def alphafold_predict_run(args):

    pf = platform.system()
    if pf == "Windows":
        print("ColabFold on Windows")
    elif pf == "Darwin":
        print("ColabFold on Mac")
        device = "cpu"
    elif pf == "Linux":
        print("ColabFold on Linux")
        device = "gpu"

    TMP_DIR = "tmp"
    os.makedirs(TMP_DIR, exist_ok=True)

    IN_COLAB = False

    print("Input ID: {}".format(readfastafile(args.input)[0]))
    print("Seq Name: {}".format(args.seq_name))
    print("Input Sequence: {}".format(readfastafile(args.input)[1]))
    sequence = str(readfastafile(args.input)[1])
    jobname = "test"  # @param {type:"string"}
    homooligomer = args.homooligomer  # @param {type:"string"}

    TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]"

    if args.output_dir != "":
        output_dir = args.output_dir
    # --set the output directory from command-line arguments

    I = cf_af.prep_inputs(
        sequence, jobname, homooligomer, output_dir, clean=IN_COLAB
    )

    msa_method = args.msa_method  # @param ["mmseqs2","single_sequence"]

    if msa_method == "precomputed":
        if args.precomputed is None:
            raise ValueError(
                "ERROR: `--precomputed` undefined. "
                "You must specify the file path of previously generated 'msa.pickle' if you set '--msa_method precomputed'."
            )
        else:
            precomputed = args.precomputed
            print("Use precomputed msa.pickle: {}".format(precomputed))
    else:
        precomputed = args.precomputed

    add_custom_msa = args.add_msa  # @param {type:"boolean"}
    c_msa = args.c_msa
    msa_format = "fas"  # @param ["fas","a2m","a3m","sto","psi","clu"]

    # --set the output directory from command-line arguments
    pair_mode = (
        args.pair_mode
    )  # @param ["unpaired","unpaired+paired","paired"] {type:"string"}
    pair_cov = args.pair_cov  # @param [0,25,50,75,90] {type:"raw"}
    pair_qid = args.pair_qid  # @param [0,15,20,30,40,50] {type:"raw"}
    # --set the output directory from command-line arguments

    # --- Search against genetic databases ---

    I = cf_af.prep_msa(
        I,
        msa_method,
        add_custom_msa,
        msa_format,
        pair_mode,
        pair_cov,
        pair_qid,
        custom_msa=c_msa,
        hhfilter_loc=HHFILTER_PATH,
        precomputed=precomputed,
        TMP_DIR=output_dir,
    )
    mod_I = I

    print("Depth", len(I["msas"][0]))

    if len(I["msas"][0]) > 1:
        plt = cf.plot_msas(I["msas"], I["ori_sequence"])
        plt.savefig(
            os.path.join(I["output_dir"], "msa_coverage.png"),
            bbox_inches="tight",
            dpi=200,
        )

    trim = ""  # @param {type:"string"}
    trim_inverse = False  # @param {type:"boolean"}
    cov = args.seqcov  # @param [0,25,50,75,90,95] {type:"raw"}
    qid = args.seqqid  # @param [0,15,20,25,30,40,50] {type:"raw"}

    mod_I = cf_af.prep_filter(I, trim, trim_inverse, cov, qid)
    print("seqcov: ", cov, "seqqid:", qid)
    print("Filtered Depth", len(mod_I["msas"][0]))

    print("Saving filtered msa...")
    pickle.dump(
        {
            "msas": mod_I["msas"],
            "deletion_matrices": mod_I["deletion_matrices"],
        },
        open(os.path.join(I["output_dir"], "msa_filtered.pickle"), "wb"),
    )
    print("Done.")

    if I["msas"] != mod_I["msas"]:
        plt.figure(figsize=(16, 5), dpi=100)
        plt.subplot(1, 2, 1)
        plt.title("Sequence coverage (Before)")
        cf.plot_msas(I["msas"], I["ori_sequence"], return_plt=False)
        plt.subplot(1, 2, 2)
        plt.title("Sequence coverage (After)")
        cf.plot_msas(mod_I["msas"], mod_I["ori_sequence"], return_plt=False)
        plt.savefig(
            os.path.join(I["output_dir"], "msa_coverage.filtered.png"),
            bbox_inches="tight",
            dpi=200,
        )
        plt.show()

    num_relax = args.num_relax
    rank_by = args.rank_by

    use_turbo = True if args.use_turbo else False
    max_msa = args.max_msa

    max_msa_clusters, max_extra_msa = [int(x) for x in max_msa.split(":")]

    show_images = True  # @param {type:"boolean"}

    num_models = args.num_models
    print(num_models)
    use_ptm = True if args.use_ptm else False
    num_ensemble = args.num_ensemble
    max_recycles = args.max_recycles
    tol = args.tol
    is_training = True if args.is_training else False
    num_samples = args.num_samples
    # --------set parameters from command-line arguments--------

    subsample_msa = True  # @param {type:"boolean"}

    if not use_ptm and rank_by == "pTMscore":
        print(
            "WARNING: models will be ranked by pLDDT, 'use_ptm' is needed to compute pTMscore"
        )
        rank_by = "pLDDT"

    # prep input features
    feature_dict = cf_af.prep_feats(mod_I, clean=IN_COLAB)
    Ls_plot = feature_dict["Ls"]

    # prep model options
    opt = {
        "N": len(feature_dict["msa"]),
        "L": len(feature_dict["residue_index"]),
        "use_ptm": use_ptm,
        "use_turbo": use_turbo,
        "max_recycles": max_recycles,
        "tol": tol,
        "num_ensemble": num_ensemble,
        "max_msa_clusters": max_msa_clusters,
        "max_extra_msa": max_extra_msa,
        "is_training": is_training,
    }

    if use_turbo:
        runner = cf_af.prep_model_runner(opt)
    else:
        runner = None

    ###########################
    # run alphafold
    ###########################
    outs, model_rank = cf_af.run_alphafold(
        feature_dict,
        opt,
        runner,
        num_models,
        num_samples,
        subsample_msa,
        params_loc=args.params_loc,
        rank_by=rank_by,
        show_images=show_images,
    )

    path_outs = os.path.join(I["output_dir"], "outs.pickle")
    print(f"writing result to {path_outs}...")
    with open(path_outs, "wb") as f:
        pickle.dump(
            {
                "outs": {
                    name: {
                        key: val
                        for key, val in res.items()
                        if key != "unrelaxed_protein"
                    }
                    for name, res in outs.items()
                },
                "model_rank": model_rank,
                "opt": opt,
            },
            f,
        )
    print("Done.")

    # @title Refine structures with Amber-Relax (Optional)

    # --------set parameters from command-line arguments--------
    num_relax = args.num_relax
    # --------set parameters from command-line arguments--------

    if num_relax == "None":
        num_relax = 0
    elif num_relax == "Top1":
        num_relax = 1
    elif num_relax == "Top5":
        num_relax = 5
    else:
        model_names = ["model_1", "model_2", "model_3", "model_4", "model_5"][
            :num_models
        ]
        # model_names = ['model_5'][:num_models]
        num_relax = len(model_names) * num_samples

    if num_relax > 0:
        if "relax" not in dir():
            # add conda environment to path
            # sys.path.append("./colabfold-conda/lib/python3.7/site-packages")

            # import libraries
            # source activate ~/localcolabfold/colabfold/colabfold-conda # absolute path of the enviroment
            # conda install -c conda-forge openmm-setup
            # vim ~/localcolabfold/colabfold/alphafold/relax/cleanup.py # change simtk.openmm to openmm, or copy revised relax directory
            from pipeline.tool.colabfold.alphafold.relax import relax
            from pipeline.tool.colabfold.alphafold.relax import utils

        with tqdm.tqdm(total=num_relax, bar_format=TQDM_BAR_FORMAT) as pbar:
            pbar.set_description(f"AMBER relaxation")
            for n, key in enumerate(model_rank):
                if n < num_relax:
                    prefix = f"rank_{n+1}_{key}"
                    pred_output_path = os.path.join(
                        I["output_dir"], f"{prefix}_relaxed.pdb"
                    )
                    if not os.path.isfile(pred_output_path):
                        amber_relaxer = relax.AmberRelaxation(
                            max_iterations=0,
                            tolerance=2.39,
                            stiffness=10.0,
                            exclude_residues=[],
                            max_outer_iterations=20,
                        )
                        relaxed_pdb_lines, _, _ = amber_relaxer.process(
                            prot=outs[key]["unrelaxed_protein"]
                        )
                        with open(pred_output_path, "w") as f:
                            f.write(relaxed_pdb_lines)
                    pbar.update(n=1)
    # @title Display 3D structure {run: "auto"}
    rank_num = 1  # @param ["1", "2", "3", "4", "5"] {type:"raw"}
    color = "lDDT"  # @param ["chain", "lDDT", "rainbow"]
    show_sidechains = False  # @param {type:"boolean"}
    show_mainchains = False  # @param {type:"boolean"}

    key = model_rank[rank_num - 1]
    prefix = f"rank_{rank_num}_{key}"
    pred_output_path = os.path.join(I["output_dir"], f"{prefix}_relaxed.pdb")
    if not os.path.isfile(pred_output_path):
        pred_output_path = os.path.join(
            I["output_dir"], f"{prefix}_unrelaxed.pdb"
        )

    cf.show_pdb(
        pred_output_path, show_sidechains, show_mainchains, color, Ls=Ls_plot
    ).show()
    if color == "lDDT":
        cf.plot_plddt_legend().show()
    if use_ptm:
        cf.plot_confidence(
            outs[key]["plddt"], outs[key]["pae"], Ls=Ls_plot
        ).show()
    else:
        cf.plot_confidence(outs[key]["plddt"], Ls=Ls_plot).show()
    #%%
    # @title Extra outputs
    dpi = 300  # @param {type:"integer"}
    save_to_txt = True  # @param {type:"boolean"}
    save_pae_json = True  # @param {type:"boolean"}

    if use_ptm:
        print("predicted alignment error")
        cf.plot_paes([outs[k]["pae"] for k in model_rank], Ls=Ls_plot, dpi=dpi)
        plt.savefig(
            os.path.join(I["output_dir"], f"predicted_alignment_error.png"),
            bbox_inches="tight",
            dpi=np.maximum(200, dpi),
        )
        plt.show()

    print("predicted contacts")
    cf.plot_adjs([outs[k]["adj"] for k in model_rank], Ls=Ls_plot, dpi=dpi)
    plt.savefig(
        os.path.join(I["output_dir"], f"predicted_contacts.png"),
        bbox_inches="tight",
        dpi=np.maximum(200, dpi),
    )
    plt.show()

    print("predicted distogram")
    cf.plot_dists([outs[k]["dists"] for k in model_rank], Ls=Ls_plot, dpi=dpi)
    plt.savefig(
        os.path.join(I["output_dir"], f"predicted_distogram.png"),
        bbox_inches="tight",
        dpi=np.maximum(200, dpi),
    )
    plt.show()

    print("predicted LDDT")
    cf.plot_plddts([outs[k]["plddt"] for k in model_rank], Ls=Ls_plot, dpi=dpi)
    plt.savefig(
        os.path.join(I["output_dir"], f"predicted_LDDT.png"),
        bbox_inches="tight",
        dpi=np.maximum(200, dpi),
    )
    plt.show()

    for n, key in enumerate(model_rank):
        if save_to_txt:
            txt_filename = os.path.join(
                I["output_dir"], f"rank_{n+1}_{key}.raw.txt"
            )
            do_save_to_txt(
                txt_filename,
                outs[key]["adj"],
                outs[key]["dists"],
                mod_I["full_sequence"],
            )

        if use_ptm and save_pae_json:
            pae = outs[key]["pae"]
            max_pae = pae.max()
            # Save pLDDT and predicted aligned error (if it exists)
            pae_output_path = os.path.join(
                I["output_dir"], f"rank_{n+1}_{key}_pae.json"
            )
            # Save predicted aligned error in the same format as the AF EMBL DB
            rounded_errors = np.round(np.asarray(pae), decimals=1)
            indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
            indices_1 = indices[0].flatten().tolist()
            indices_2 = indices[1].flatten().tolist()
            pae_data = json.dumps(
                [
                    {
                        "residue1": indices_1,
                        "residue2": indices_2,
                        "distance": rounded_errors.flatten().tolist(),
                        "max_predicted_aligned_error": max_pae.item(),
                    }
                ],
                indent=None,
                separators=(",", ":"),
            )
            with open(pae_output_path, "w") as f:
                f.write(pae_data)

        result_data = json.dumps(
            {"plddt": outs[key]["pLDDT"].tolist()}, indent=None
        )

        result_prefix = key[:7]
        save_results(I["output_dir"], result_prefix, result_data)

        if n == 0:
            result_prefix = "model_best"
            save_results(I["output_dir"], result_prefix, result_data)

    print("seq_name:", args.seq_name)
    print("Custom MSA:", args.c_msa)
    print("Depth", len(I["msas"][0]))
    print("seqcov: ", cov, "seqqid:", qid)
    print("Filtered Depth", len(mod_I["msas"][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runner script that can take command-line arguments"
    )
    parser.add_argument(
        "--params_loc", default="/sharefs/thuhc-data/alphafold/"
    )
    parser.add_argument(
        "-i", "--input", help="Path to a FASTA file. Required.", required=True
    )
    parser.add_argument(
        "-cm",
        "--c_msa",
        help="Path to a custom msa FASTA file. Required.",
        required=True,
    )
    parser.add_argument(
        "-a_m",
        "--add_msa",
        type=bool,
        help="whether we need to add msa/use custom msa",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="",
        type=str,
        help="Path to a directory that will store the results. "
        "The default name is 'prediction_<hash>'. ",
    )
    parser.add_argument(
        "-sc",
        "--seqcov",
        default="0",
        type=float,
        help="the coverage filter for msa",
    )
    parser.add_argument(
        "-sq",
        "--seqqid",
        default="0",
        type=float,
        help="the indentity filter for msa",
    )
    parser.add_argument(
        "-seqn",
        "--seq_name",
        default="",
        type=str,
        help="the symbol of sequence",
    )
    parser.add_argument(
        "-ho",
        "--homooligomer",
        default="1",
        type=str,
        help="homooligomer: Define number of copies in a homo-oligomeric assembly. "
        "For example, sequence:ABC:DEF, homooligomer: 2:1, "
        "the first protein ABC will be modeled as a homodimer (2 copies) and second DEF a monomer (1 copy). Default is 1.",
    )
    parser.add_argument(
        "-m",
        "--msa_method",
        default="single_sequence",
        type=str,
        choices=["mmseqs2", "single_sequence", "precomputed"],
        help="Options to generate MSA."
        "mmseqs2 - FAST method from ColabFold (default) "
        "single_sequence - use single sequence input."
        "precomputed - specify 'msa.pickle' file generated previously if you have."
        "Default is 'mmseqs2'.",
    )
    parser.add_argument(
        "--precomputed",
        default=None,
        type=str,
        help="Specify the file path of a precomputed pickled msa from previous run. ",
    )
    parser.add_argument(
        "-p",
        "--pair_mode",
        default="unpaired",
        choices=["unpaired", "unpaired+paired", "paired"],
        help="Experimental option for protein complexes. "
        "Pairing currently only supported for proteins in same operon (prokaryotic genomes). "
        "unpaired - generate separate MSA for each protein. (default) "
        "unpaired+paired - attempt to pair sequences from the same operon within the genome. "
        "paired - only use sequences that were successfully paired. "
        "Default is 'unpaired'.",
    )
    parser.add_argument(
        "-pc",
        "--pair_cov",
        default=50,
        type=int,
        help="Options to prefilter each MSA before pairing. It might help if there are any paralogs in the complex. "
        "prefilter each MSA to minimum coverage with query (%%) before pairing. "
        "Default is 50.",
    )
    parser.add_argument(
        "-pq",
        "--pair_qid",
        default=20,
        type=int,
        help="Options to prefilter each MSA before pairing. It might help if there are any paralogs in the complex. "
        "prefilter each MSA to minimum sequence identity with query (%%) before pairing. "
        "Default is 20.",
    )
    parser.add_argument(
        "-b",
        "--rank_by",
        default="pLDDT",
        type=str,
        choices=["pLDDT", "pTMscore"],
        help="specify metric to use for ranking models (For protein-protein complexes, we recommend pTMscore). "
        "Default is 'pLDDT'.",
    )
    parser.add_argument(
        "-t",
        "--use_turbo",
        action="store_true",
        help="introduces a few modifications (compile once, swap params, adjust max_msa) to speedup and reduce memory requirements. "
        "Disable for default behavior.",
    )
    parser.add_argument(
        "-mm",
        "--max_msa",
        default="512:1024",
        type=str,
        help="max_msa defines: max_msa_clusters:max_extra_msa number of sequences to use. "
        "This option ignored if use_turbo is disabled. Default is '512:1024'.",
    )
    parser.add_argument(
        "-n",
        "--num_models",
        default=5,
        type=int,
        help="specify how many model params to try. (Default is 5)",
    )
    parser.add_argument(
        "-pt",
        "--use_ptm",
        action="store_true",
        help="uses Deepmind's ptm finetuned model parameters to get PAE per structure. "
        "Disable to use the original model params. (Disabling may give alternative structures.)",
    )
    parser.add_argument(
        "-e",
        "--num_ensemble",
        default=1,
        type=int,
        choices=[1, 8],
        help="the trunk of the network is run multiple times with different random choices for the MSA cluster centers. "
        "(1=default, 8=casp14 setting)",
    )
    parser.add_argument(
        "-r",
        "--max_recycles",
        default=3,
        type=int,
        help="controls the maximum number of times the structure is fed back into the neural network for refinement. (default is 3)",
    )
    parser.add_argument(
        "--tol",
        default=0,
        type=float,
        help="tolerance for deciding when to stop (CA-RMS between recycles)",
    )
    parser.add_argument(
        "--is_training",
        action="store_true",
        help="enables the stochastic part of the model (dropout), when coupled with num_samples can be used to 'sample' a diverse set of structures. False (NOT specifying this option) is recommended at first.",
    )
    parser.add_argument(
        "--num_samples",
        default=1,
        type=int,
        help="number of random_seeds to try. Default is 1.",
    )
    parser.add_argument(
        "--num_relax",
        default="Top5",
        choices=["None", "Top1", "Top5", "All"],
        help="num_relax is 'None' (default), 'Top1', 'Top5' or 'All'. Specify how many of the top ranked structures to relax.",
    )
    args = parser.parse_args()
    alphafold_predict_run(args)
