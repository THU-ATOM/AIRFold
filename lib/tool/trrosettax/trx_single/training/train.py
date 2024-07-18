import warnings
import os
import random
import sys
import torch

import numpy as np
from torch.utils.data import DataLoader
from time import time
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from einops.layers.torch import Rearrange

base_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent)
sys.path.insert(0, base_dir)
sys.path.insert(1, f'{base_dir}/trx_single')
from models.res2net import *
from models.res2net_MSA import DistPredictorMSA
from models.modules import empty_cache
from trx_single.utils.utils_data import get_clusters, save_to_json, bcolors
from trx_single.utils.dataset import SingleDataset
from trx_single.utils.utils_training import load_pretrain, geometry_loss, distill_loss

parser = ArgumentParser(description='Training of trRosettaX-Single', add_help=False, formatter_class=RawTextHelpFormatter)
group_help = parser.add_argument_group('help information')
group_help.add_argument("-h", "--help", action="help", help="show this help message and exit")

# common arguments
group = parser.add_argument_group('common arguments')
group.add_argument('npz_dir',
                   type=str,
                   help=f'{bcolors.BOLD}(required) Location of folder storing npz files storing sequence and labels parsing from PDB files{bcolors.RESET}')
group.add_argument('ckpt_dir',
                   type=str,
                   help=f"{bcolors.BOLD}(required) Location of folder to save checkpoints to{bcolors.RESET}")
group.add_argument('esm1b',
                   type=str,
                   help=f'{bcolors.BOLD}(required) Location of the pretrained ESM-1b{bcolors.RESET}\n' + 'download from ' + f'{bcolors.UNDERLINE}{bcolors.BLUE}https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt{bcolors.RESET}')

group.add_argument('-s',
                   '--train_stage',
                   type=int, choices=[1, 2],
                   help=f'{bcolors.BOLD}(required) Stage of training\n'
                        f'1:knowledge distillation\n'
                        f'2:supervised ESM-1b {bcolors.RESET}')
group.add_argument('-init_lr', '--init_lr',
                   type=float, default=1e-4,
                   help='Initial learning rate (default:0.0001)')
group.add_argument('-batch_size', '--batch_size',
                   type=int, default=16,
                   help='Batch size for training (default:16)')
group.add_argument('-early_stopping', '--early_stopping',
                   action='store_true', default=True,
                   help='Whether to stop early if val loss cannot drop for several epochs (default:True)')
group.add_argument('-max_epochs', '--max_epochs',
                   type=int, default=30,
                   help='Maximum number of epochs to train (default:30)')
group.add_argument('-crop_size', '--crop_size',
                   type=int, default=200,
                   help='Sequence with residues more than this number will be randomly cropped for training (default:300)')
group.add_argument('-gpu', '--gpu',
                   type=int, default=0,
                   help='Use which gpu. cpu only if set to -1 (default:0)')
group.add_argument('-cpu', '--cpu',
                   type=int, default=2,
                   help='Number of cpus to use (default:2)')
group.add_argument('-silent', '--silent',
                   action='store_true',
                   help='Run in silent mode (default: False)')

# stage 1 arguments
group1 = parser.add_argument_group('for stage 1 (i.e., knowledge distillation)')
group1.add_argument('-t',
                    '--teacher',
                    type=str, default=f'{base_dir}/trx_single/training/materials/res2net_MSA.pth.tar',
                    help='Path to the parameters of Res2Net_MSA (default: trx_single/teacher.pth.tar)')

# stage 2 arguments
group2 = parser.add_argument_group('for stage 2 (i.e., supervised re-training of ESM-1b)')
group2.add_argument('-init_model', '--init_model',
                    type=str, default=None,
                    help='(required) Path to the distilled parameters of Res2Net_Single')
group2.add_argument('-clstr', '--clstr',
                    type=str, default=None,
                    help='(required) File storing the cluster information of training set (by cd-hit).\n'
                         'If not assigned, all samples will be used in a single epoch rather than randomly selected from each cluster')
group2.add_argument('-mask', '--mask',
                    type=float, default=.15,
                    help='Sequence mask rate (default:0.15)')

group3 = parser.add_argument_group('optional arguments')


def check_args(args):
    if args.train_stage == 1:
        assert os.path.isfile(args.teacher), f'{bcolors.FAIL}Please provide the teacher model to guide the distillation in stage 1!{bcolors.RESET}'
    else:
        if args.init_model is None:
            warnings.warn(f'{bcolors.WARNING}No pretrained parameters are provided for Res2Net_Single. The stage 2 will start from the randomly initialized parameters.{bcolors.RESET}')
        if args.clstr is None:
            warnings.warn(f'{bcolors.WARNING}No cluster information is provided. All the samples in {args.npz_dir} will be used in a single epoch.{bcolors.RESET}')


def train(dataloader, is_training=False):
    model.train(is_training)
    esm_net.train(is_training)
    losses = {'total': [], 'geometry': [], 'distill': [], 'mask': []}
    n_OOM = 0

    for i, data in enumerate(dataloader):
        if not data: continue
        seq = data['seq'].long().to(device)
        seq_esm = data['seq_esm'].long().to(device)

        try:
            ## generate embeddings by ESM-1b
            with torch.set_grad_enabled(is_training and args.train_stage == 2):
                attentions, hidden_representation, logits = esm_net(seq_esm, need_head_weights=True)
                empty_cache()
                emb_out = {
                    'attentions': Rearrange('l h m n -> m n (l h)')(attentions.squeeze(0)[..., 1:-1, 1:-1]),
                    'representations': hidden_representation.squeeze(0)[1:-1],
                    'logits': logits.squeeze(0)[1:-1]
                }

            ## generate soft labels by Res2Net_MSA
            if args.train_stage == 1:
                if 'msa' not in data:
                    warn = f'{bcolors.RED} msa is needed for distillation but no msa is found in features!{bcolors.RESET}'
                    warnings.warn(warn)
                else:
                    with torch.no_grad():
                        msa = data['msa'].to(device)
                        pred_geoms_msa = model_msa(msa)

            ## predict
            if args.train_stage == 1:
                pred_geoms = model(seq, emb_out, is_training=is_training)
            else:
                pred_geoms, pred_seq = model(seq, emb_out, mask=args.mask, is_training=is_training)

            ## losses
            L_geometry = geometry_loss(pred_geoms, data, device=device)
            if args.train_stage == 1:
                L_distill = distill_loss(pred_geoms, pred_geoms_msa, device=device)
                loss = L_geometry + L_distill
            if args.train_stage == 2:
                seq_onehot = (torch.arange(33).to(device) == seq_esm[..., 1:-1, None]).float()
                L_mask = -(torch.log(pred_seq.softmax(-1) + 1e-8) * seq_onehot).sum(-1).mean()
                loss = L_geometry + L_mask

            loss_np = float(loss.detach())
            if np.isnan(loss_np):
                warn = f'{bcolors.RED}loss is nan!{bcolors.RESET}'
                warnings.warn(warn)
                continue

            if is_training:
                loss.backward()
                if i % args.batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            losses['total'].append(loss_np)
            losses['geometry'].append(L_geometry.detach().cpu().numpy())
            if args.train_stage == 1:
                losses['distill'].append(L_distill.detach().cpu().numpy())
            if args.train_stage == 2:
                losses['mask'].append(L_mask.detach().cpu().numpy())

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                n_OOM += 1
                if not args.silent:
                    warn = f'{bcolors.RED}OOM for sequence with {seq.size(-1)} residues!{bcolors.RESET}'
                    warnings.warn(warn)
                    empty_cache()
                continue
            else:
                raise exception

        if not args.silent:
            sample_name = 'train' if is_training else 'val'
            total = len(train_set) if is_training else len(val_set)
            if args.train_stage == 1:
                print(
                    f"\rEpoch: {epoch + 1}, {sample_name}: {i + 1}/{total}, loss: {loss_np:.2f}, L_geometry: {L_geometry.detach().cpu().numpy():.2f}, L_distill: {L_distill.detach().cpu().numpy():.2f}   ",
                    end='')
            else:
                print(
                    f"\rEpoch: {epoch + 1}, {sample_name}: {i + 1}/{total}, loss: {loss_np:.2f}, L_geometry: {L_geometry.detach().cpu().numpy():.2f}, L_mask: {L_mask.detach().cpu().numpy():.2f}   ",
                    end='')
    if not args.silent:
        print('\r' + ' ' * 80)
    return losses, n_OOM


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from trx_single import esm_jit

    torch.set_num_threads(args.cpu)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    checkpoint_path = f'{args.ckpt_dir}/ckpt'
    log_file = f'{args.ckpt_dir}/training_stage{args.train_stage}.log'
    os.makedirs(checkpoint_path, exist_ok=True)

    # Prepare train and validation datasets
    if args.clstr is not None:
        ## parse clstr file
        clusters = get_clusters(args.clstr)
        val_idx = random.sample(list(range(len(clusters))), 500)
        with open(f'{args.ckpt_dir}/val_clstr.lst', 'w') as f:
            f.write('\n'.join([str(idx) for idx in val_idx]))
        train_idx = list(set(range(len(clusters))) - set(val_idx))
        val_lst = []
        for i in val_idx:
            for pid in clusters[i]:
                if os.path.isfile(f'{args.npz_dir}/{pid}.npz'):
                    val_lst.append(pid)
                    break
        train_set = SingleDataset(targets=train_idx,
                                  root_dir=args.npz_dir,
                                  lengthmax=args.crop_size,
                                  clusters=clusters,
                                  return_msa=args.train_stage == 1,
                                  warning=not args.silent)
    else:
        files_lst = [f.split('.')[0] for f in os.listdir(args.npz_dir)]
        val_lst = random.sample(files_lst, 752)
        with open(f'{args.ckpt_dir}/val.lst', 'w') as f:
            f.write('\n'.join(val_lst))
        val_pid = [f[:4] for f in val_lst]
        train_lst = [f for f in files_lst if f[:4] not in val_pid]

        train_set = SingleDataset(targets=train_lst,
                                  root_dir=args.npz_dir,
                                  lengthmax=args.crop_size,
                                  clusters=None,
                                  return_msa=args.train_stage == 1,
                                  warning=not args.silent)

    val_set = SingleDataset(targets=val_lst,
                            root_dir=args.npz_dir,
                            lengthmax=args.crop_size,
                            clusters=None,
                            return_msa=args.train_stage == 1,
                            warning=not args.silent)

    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=args.cpu)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=args.cpu)

    if not args.silent:
        print(f'\n{bcolors.BLUE}train_set:{len(train_set)}{bcolors.RESET}')
        print(f'{bcolors.BLUE}val_set:{len(val_set)}{bcolors.RESET}')
        print(f'\n{bcolors.BOLD}total_epochs:{args.max_epochs}, learning_rate:{args.init_lr}\n')

    # Define network models
    ## initialize Res2Net_Single
    if not args.silent:
        if args.init_model is not None:
            print(f'{bcolors.GREEN}initialize a Res2Net_Single model from {args.init_model}{bcolors.RESET}')
        else:
            print(f'{bcolors.GREEN}initialize a Res2Net_Single model{bcolors.RESET}')
    model = DistPredictorSeq().to(device)
    if args.init_model is not None:
        model = load_pretrain(model, pt=args.init_model, device=device)
    model.train()

    ## load teacher model
    if args.train_stage == 1:
        if not args.silent:
            print(f'{bcolors.GREEN}load teacher model from {args.teacher}{bcolors.RESET}')
        model_msa = DistPredictorMSA().cuda()
        model_msa.load_state_dict(torch.load(args.teacher)['state_dict'])
        model_msa.eval()

    ## load ESM-1b
    if not args.silent:
        print(f'{bcolors.GREEN}load ESM-1b{bcolors.RESET}')
    esm_net, alphabet = esm_jit.pretrained.load_model_and_alphabet_local(args.esm1b, cuda_lst=None)
    esm_net.to(device)
    esm_net.train(args.train_stage == 1)

    # Define optimizer and scheduler
    params_nodecay = []
    params_decay = []
    for name, param in model.named_parameters():
        if 'bias' in name or 'bn' in name or 'norm' in name:
            params_nodecay.append(param)
        elif 'conv' in name or 'weight' in name:
            params_decay.append(param)
        else:
            raise NameError(f'param {name} is neither weight nor bias!')

    params = [
        {'params': params_decay, 'weight_decay': 1e-4},
        {'params': params_nodecay, 'weight_decay': 0.}
    ]

    if args.train_stage == 2:
        params.append({'params': esm_net.parameters(), 'lr': 1e-5})

    optimizer = torch.optim.Adam(params, lr=args.init_lr)
    if args.train_stage == 1:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 18, 20, 25], gamma=.6, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.9, last_epoch=-1)

    min_loss = np.inf
    steps_without_enhancing = 0
    endurance = 5

    # training
    print(f'\n{bcolors.BOLD}{bcolors.BLUE}--------------------train start-------------------------{bcolors.RESET}', end='\n')
    for epoch in range(0, args.max_epochs):

        train_losses, n_OOM_train = train(train_dataloader, is_training=True)
        val_losses, n_OOM_val = train(val_dataloader, is_training=False)
        scheduler.step()

        ## output log
        if args.train_stage == 1:
            log_info = f'\r{bcolors.BOLD}{bcolors.HEADER}---------------------Epoch: {epoch + 1}----------------------{bcolors.RESET}\n' + \
                       f'train loss: {np.mean(train_losses["total"]):.2f}, L_geometry: {np.mean(train_losses["geometry"]):.2f}, L_distill: {np.mean(train_losses["distill"]):.2f}\n' \
                       f'val loss: {np.mean(val_losses["total"]):.2f}, L_geometry: {np.mean(val_losses["geometry"]):.2f}, L_distill: {np.mean(val_losses["distill"]):.2f}{bcolors.RESET}\n'
        else:
            log_info = f'\r{bcolors.BOLD}{bcolors.HEADER}---------------------Epoch: {epoch + 1}----------------------{bcolors.RESET}\n' + \
                       f'train loss: {np.mean(train_losses["total"]):.2f}, L_geometry: {np.mean(train_losses["geometry"]):.2f}, L_mask: {np.mean(train_losses["mask"]):.2f}\n' \
                       f'val loss: {np.mean(val_losses["total"]):.2f}, L_geometry: {np.mean(val_losses["geometry"]):.2f}, L_mask: {np.mean(val_losses["mask"]):.2f}{bcolors.RESET}\n'

        if not args.silent:
            print(log_info)
        with open(log_file, 'a') as f:
            f.write(log_info.replace(bcolors.CYAN, '').replace(bcolors.BOLD, '').replace(bcolors.RESET, '').replace(bcolors.HEADER, ''))

        ## save if achieve lower validation loss
        if np.mean(val_losses['total']) < min_loss:
            steps_without_enhancing = 0
            min_loss = np.mean(val_losses['total'])
            ckpt_dict = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(ckpt_dict, f'{checkpoint_path}/res2net_stage{args.train_stage}.pth.tar')
            if args.train_stage == 2:
                model_scripted = torch.jit.script(esm_net)  # Export to TorchScript
                model_scripted.save(f'{checkpoint_path}/sESM-1b.pt')
                # esm_dict = {'epoch': epoch, 'state_dict': esm_net.state_dict()}
                # torch.save(esm_dict, f'{checkpoint_path}/sESM-1b.pth.tar')
            device_dict = dict((ind, int(str(optimizer.state_dict()['state'][ind]['exp_avg'].device)[-1])) for ind in optimizer.state_dict()['state'])
            save_to_json(device_dict, f'{checkpoint_path}/opt_devices.json')
        elif epoch > 10:
            steps_without_enhancing += 1
            if steps_without_enhancing == endurance:
                log_info = f'{bcolors.RED}-------------------early stopped!----------------------{bcolors.RESET}'
                if not args.silent:
                    print(log_info)
                with open(log_file, 'a') as f:
                    f.write(log_info.replace(bcolors.RED, '').replace(bcolors.RESET, ''))
                break

    print(f'{bcolors.BOLD}{bcolors.BLUE}---------------------train end-------------------------{bcolors.RESET}')
