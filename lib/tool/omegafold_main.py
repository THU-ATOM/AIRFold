import gc
import logging
import os
import sys
import time
import argparse
import collections
import logging
import os.path
import typing
import torch
from torch import hub

import omegafold as of
from lib.tool.omegafold import pipeline
from lib.utils.systool import get_available_gpus


def _load_weights(
        weights_url: str, weights_file: str,
) -> collections.OrderedDict:
    """
    Loads the weights from either a url or a local file. If from url,

    Args:
        weights_url: a url for the weights
        weights_file: a local file

    Returns:
        state_dict: the state dict for the model

    """

    weights_file = os.path.expanduser(weights_file)
    use_cache = os.path.exists(weights_file)
    if weights_file and weights_url and not use_cache:
        logging.info(
            f"Downloading weights from {weights_url} to {weights_file}"
        )
        os.makedirs(os.path.dirname(weights_file), exist_ok=True)
        hub.download_url_to_file(weights_url, weights_file)
    else:
        logging.info(f"Loading weights from {weights_file}")

    return torch.load(weights_file, map_location='cpu')


def get_args() -> typing.Tuple[
    argparse.Namespace, collections.OrderedDict, argparse.Namespace]:
    """
    Parse the arguments, which includes loading the weights

    Returns:
        input_file: the path to the FASTA file to load sequences from.
        output_dir: the output folder directory in which the PDBs will reside.
        batch_size: the batch_size of each forward
        weights: the state dict of the model

    """
    parser = argparse.ArgumentParser(
        description=
        """
        Launch OmegaFold and perform inference on the data. 
        Some examples (both the input and output files) are included in the 
        Examples folder, where each folder contains the output of each 
        available model from model1 to model3. All of the results are obtained 
        by issuing the general command with only model number chosen (1-3).
        """
    )
    parser.add_argument(
        '--input_file', type=lambda x: os.path.expanduser(str(x)),
        help=
        """
        The input fasta file
        """
    )
    parser.add_argument(
        '--output_file', type=lambda x: os.path.expanduser(str(x)),
        help=
        """
        The output pdb file
        """
    )
    parser.add_argument(
        '--output_dir', type=lambda x: os.path.expanduser(str(x)),
        help=
        """
        The output directory to write the output pdb files. 
        If the directory does not exist, we just create it. 
        The output file name follows its unique identifier in the 
        rows of the input fasta file"
        """
    )
    parser.add_argument(
        '--num_cycle', default=10, type=int,
        help="The number of cycles for optimization, default to 10"
    )
    parser.add_argument(
        '--subbatch_size', default=None, type=int,
        help=
        """
        The subbatching number, 
        the smaller, the slower, the less GRAM requirements. 
        Default is the entire length of the sequence.
        This one takes priority over the automatically determined one for 
        the sequences
        """
    )
    parser.add_argument(
        '--weights_file',
        default="/data/protein/datasets_2024/OmegaFold/release1.pt",
        type=str,
        help='The model cache to run, default os.path.expanduser("~/.cache/omegafold_ckpt/model.pt")'
    )
    parser.add_argument(
        '--weights',
        default="https://helixon.s3.amazonaws.com/release1.pt",
        type=str,
        help='The url to the weights of the model'
    )
    parser.add_argument(
        '--model', default=1, type=int,
        help='The model number to run, current we support 1 or 2'
    )
    parser.add_argument(
        '--pseudo_msa_mask_rate', default=0.12, type=float,
        help='The masking rate for generating pseudo MSAs'
    )
    parser.add_argument(
        '--num_pseudo_msa', default=15, type=int,
        help='The number of pseudo MSAs'
    )

    args = parser.parse_args()

    if args.model == 1:
        weights_url = "https://helixon.s3.amazonaws.com/release1.pt"
        if args.weights_file is None:
            args.weights_file = os.path.expanduser(
                "~/.cache/omegafold_ckpt/model.pt"
            )
    elif args.model == 2:
        weights_url = "https://helixon.s3.amazonaws.com/release2.pt"
        if args.weights_file is None:
            args.weights_file = os.path.expanduser(
                "~/.cache/omegafold_ckpt/model2.pt"
            )
    else:
        raise ValueError(
            f"Model {args.model} is not available, "
            f"we only support model 1 and 2"
        )
    weights_file = args.weights_file
    # if the output directory is not provided, we will create one alongside the
    # input fasta file
    if weights_file or weights_url:
        weights = _load_weights(weights_url, weights_file)
        weights = weights.pop('model', weights)
    else:
        weights = None

    forward_config = argparse.Namespace(
        subbatch_size=args.subbatch_size,
        num_recycle=args.num_cycle,
    )

    return args, weights, forward_config

@torch.no_grad()
def prediction():
    args, state_dict, forward_config = get_args()
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # get the model
    logging.info(f"Constructing OmegaFold")
    model = of.OmegaFold(of.make_config(args.model))
    if state_dict is None:
        logging.warning("Inferencing without loading weight")
    else:
        if "model" in state_dict:
            state_dict = state_dict.pop("model")
        model.load_state_dict(state_dict)
    model.eval()
    
    device_ids = get_available_gpus(1)
    args.device = torch.device(f"cuda:{device_ids[0]}") if torch.cuda.is_available() else 'cpu'
    model.to(args.device)

    logging.info(f"Reading {args.input_file}")
    for i, (input_data, save_path) in enumerate(
            pipeline.fasta2inputs(
                args.input_file,
                num_pseudo_msa=args.num_pseudo_msa,
                output_dir=args.output_dir,
                device=args.device,
                mask_rate=args.pseudo_msa_mask_rate,
                num_cycle=args.num_cycle,
            )
    ):
        logging.info(f"Predicting {i + 1}th chain in {args.input_file}")
        logging.info(
            f"{len(input_data[0]['p_msa'][0])} residues in this chain."
        )
        ts = time.time()
        try:
            output = model(
                    input_data,
                    predict_with_confidence=True,
                    fwd_cfg=forward_config
                )
        except RuntimeError as e:
            logging.info(f"Failed to generate {save_path} due to {e}")
            logging.info(f"Skipping...")
            continue
        logging.info(f"Finished prediction in {time.time() - ts:.2f} seconds.")

        logging.info(f"Saving prediction to {save_path}")
        pipeline.save_pdb(
            pos14=output["final_atom_positions"],
            b_factors=output["confidence"] * 100,
            sequence=input_data[0]["p_msa"][0],
            mask=input_data[0]["p_msa_mask"][0],
            save_path=save_path,
            model=0
        )
        
        os.rename(save_path,args.output_file)
        
        logging.info(f"Saved")
        del output
        torch.cuda.empty_cache()
        gc.collect()
    logging.info("Done!")


if __name__ == '__main__':
    prediction()
