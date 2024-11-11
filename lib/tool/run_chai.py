import argparse
import torch
from lib.utils.systool import get_available_gpus
from  loguru import logger
from pathlib import Path

from lib.tool.chai_util.chai_pdb import run_inference
from chai_lab.data.dataset.inference_dataset import (load_chains_from_raw,
                                                     read_inputs)
from chai_lab.data.parsing.msas.aligned_pqt import (a3m_to_aligned_dataframe,
                                                    expected_basename)

def chai_main(args):

    assert Path(args.fasta_path).exists(), args.fasta_path
    fasta_inputs = read_inputs(args.fasta_path, length_limit=None)
    assert len(fasta_inputs) > 0, "No inputs found in fasta file"
        
    # a3m to expected_basename pqt fle
    chains = load_chains_from_raw(fasta_inputs)
    for i, chain in enumerate(chains):
        pqt_path = Path(args.msa_dir / expected_basename(chain.entity_data.sequence))
        df = a3m_to_aligned_dataframe(a3m_path=Path(args.a3m_paths[i]))
        df.to_parquet(pqt_path)

    # get device
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'

    run_inference(
            fasta_file=Path(args.fasta_path),
            output_dir=Path(args.output_dir),
            msa_directory=Path(args.msa_dir),
            # 'default' setup
            # We run Chai-1 with 4 recycles, 
            # 5 trunk samples and 5 diffusion samples 
            # for a total of 25 predicted structures unless otherwise noted. 
            num_trunk_recycles=args.ntr,
            num_diffn_timesteps=args.ndt,
            seed=args.random_seed,
            device=device,
            use_esm_embeddings=True,
        )


def chai_single_main(args):

    assert Path(args.fasta_path).exists(), args.fasta_path
    fasta_inputs = read_inputs(args.fasta_path, length_limit=None)
    assert len(fasta_inputs) > 0, "No inputs found in fasta file"

    # get device
    device_ids = get_available_gpus(1)
    device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'

    run_inference(
            fasta_file=Path(args.fasta_path),
            output_dir=Path(args.output_dir),
            # 'default' setup
            # We run Chai-1 with 4 recycles, 
            # 5 trunk samples and 5 diffusion samples 
            # for a total of 25 predicted structures unless otherwise noted. 
            num_trunk_recycles=args.ntr,
            num_diffn_timesteps=args.ndt,
            seed=args.random_seed,
            device=device,
            use_esm_embeddings=True,
        )


if __name__ == "__main__":
    logger.info("RUNNING ESMFOLD PREDICTION START!")
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--msa_path", type=str, required=True)
    parser.add_argument("--a3m_paths", type=str, required=True, nargs='*')
    
    parser.add_argument("--ntr", type=int, default=4)
    parser.add_argument("--ndt", type=int, default=200)
    parser.add_argument("--random_seed", type=int, default=0)
    
    argv = parser.parse_args()
    
    # get sequence
    logger.info(f"++++ RUNNING Chai-1 PREDICTION ++++")
    chai_main(argv)
