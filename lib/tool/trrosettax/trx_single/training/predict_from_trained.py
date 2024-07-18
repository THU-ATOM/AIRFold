import argparse
import numpy as np
import sys
import os
import torch
from einops.layers.torch import Rearrange
from pathlib import Path
from einops import rearrange
from collections import defaultdict

sys.path.insert(0,str(Path(os.path.abspath(os.path.dirname(__file__))).parent.parent))
import trx_single
from trx_single.models.res2net import *
from trx_single.models.modules import empty_cache
from trx_single.utils.utils_data import *

parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description='Predict based on model trained by train.py.',
                                 epilog="""An example:
python predict_from_trained.py -i ../example/seq.fasta -o ../example/seq.npz -mdir ./training/ckpt -gpu 0
                                 """)

group_help = parser.add_argument_group('help information')
group_help.add_argument("-h", "--help", action="help", help="show this help message and exit")

group = parser.add_argument_group('required arguments')
group.add_argument("-i", type=str, dest='FASTA', required=True, help="input AA sequence in FASTA format")
group.add_argument("-o", type=str, dest='NPZ', required=True, help="predicted distograms and anglegrams")
group.add_argument('-mdir', type=str, dest='MDIR', required=True, help='folder with the pre-trained network')

group1 = parser.add_argument_group('optional arguments')
group1.add_argument("-cont", dest='CONT', default=None, help="csv file storing the predicted contacts, default not to output")
group1.add_argument('-gpu', dest='GPU', default='-1', type=int, help='use which gpu. cpu only if set to -1(default:-1)')
group1.add_argument('-cpu', dest='CPU', default=2, type=int, help='number of cpus to use(default:2)')

args = parser.parse_args()

fasta_file = os.path.abspath(args.FASTA)
npz_file = args.NPZ

file_dir, file_name = os.path.split(fasta_file)
jobname = os.path.splitext(file_name)[0]

cont_file = None if args.CONT is None else str(args.CONT)
MDIR = Path(args.MDIR)

torch.set_num_threads(args.CPU)
gpu_id = args.GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
device = torch.device(f'cuda:0') if gpu_id >= 0 else torch.device('cpu')
torch._C._jit_set_profiling_mode(False)


def predict(mname, window=500, shift=300):
    global model, seq
    model_CKPT = torch.load(MDIR / f'{mname}.pth.tar', map_location='cpu')
    # recycle = bool(model_CKPT['recycle'])
    # if recycle:
    #     model_ = model_recycle
    # else:
    #     model_ = model
    model.load_state_dict(model_CKPT['state_dict'])
    model.eval()
    model = model.to(device)

    # TODO
    esm_model = torch.jit.load(str(MDIR / f'sESM-1b.pt'), map_location=device)
    with torch.no_grad():
        L = seq.shape[-1]
        idx_pdb = torch.arange(L).long().view(1, L)
        # esm-1b can only handle sequences with <1024 AAs
        # run esm-1b by crops if length > 1000
        if L > 1000:
            esm_out = {
                'attentions': torch.zeros((L, L, 660), device=device),
                'representations': torch.zeros((L, 1280), device=device),
            }
            count_1d = torch.zeros((L), device=device)
            count_2d = torch.zeros((L, L), device=device)
            #
            grids = np.arange(0, L - window + shift, shift)
            ngrids = grids.shape[0]
            print("ngrid:     ", ngrids)
            print("grids:     ", grids)
            print("windows:   ", window)

            for i in range(ngrids):
                for j in range(i, ngrids):
                    start_1 = grids[i]
                    end_1 = min(grids[i] + window, L)
                    start_2 = grids[j]
                    end_2 = min(grids[j] + window, L)
                    sel = np.zeros((L)).astype(np.bool)
                    sel[start_1:end_1] = True
                    sel[start_2:end_2] = True

                    input_seq = seq[:, sel]
                    input_seq = torch.from_numpy(mymsa_to_esmmsa(input_seq, input_type='fasta')).long().to(device)
                    input_idx = idx_pdb[:, sel]

                    print("running crop: %d-%d/%d-%d" % (start_1, end_1, start_2, end_2), input_seq.shape)
                    with torch.cuda.amp.autocast(enabled=False):
                        attentions_crop, representations_crop,_ = esm_model(input_seq)
                    empty_cache()

                    weight = 1
                    sub_idx = input_idx[0].cpu()
                    sub_idx_2d = np.ix_(sub_idx, sub_idx)
                    count_1d[sub_idx] += weight
                    count_2d[sub_idx_2d] += weight

                    esm_out['representations'][sub_idx] += weight * representations_crop.squeeze(0)[1:-1]
                    attentions_crop = attentions_crop.squeeze(0)[..., 1:-1, 1:-1]
                    attentions_crop = rearrange(attentions_crop, 'l h m n -> m n (l h)')
                    attentions_crop *= weight
                    esm_out['attentions'][sub_idx_2d] += attentions_crop
                    del representations_crop, attentions_crop
                    empty_cache()

            esm_out['representations'] /= count_1d[:, None]
            esm_out['attentions'] /= count_2d[:, :, None]
        else:
            seq_esm = torch.from_numpy(mymsa_to_esmmsa(seq, input_type='fasta')).long().to(device)
            attentions, representations,_ = esm_model(seq_esm)
            empty_cache()
            esm_out = {
                'attentions': Rearrange('l h m n -> m n (l h)')(attentions.squeeze(0)[..., 1:-1, 1:-1]),
                'representations': representations.squeeze(0)[1:-1],
            }
        pred_res = model(seq_torch, esm_out)
    return pred_res


if __name__ == '__main__':

    print('Predicting inter-residue geometries...')
    """ parse fasta file to numpy array """
    seq = parse_seq(fasta_file)
    seq_torch = torch.from_numpy(seq).long().to(device)

    """ initialize """
    model = DistPredictorSeq()
    # model_recycle = DistPredictorSeqRecycle()
    # model_recycle.emb_net = model.emb_net

    """ predict """
    pred_res_all = defaultdict(list)
    pred_res = predict('res2net_stage2')
    print(f'done')
    for k in pred_res:
        pred_res_all[k].append(pred_res[k].cpu().numpy())

    print('Ensembling and saving...')
    for k in pred_res_all:
        pred_res_all[k] = np.mean(pred_res_all[k], axis=0)

    os.makedirs(os.path.dirname(npz_file), exist_ok=True)

    np.savez_compressed(npz_file, **pred_res_all)
