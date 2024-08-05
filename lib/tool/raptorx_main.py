import os
import time

import torch
from torch.utils.data import DataLoader

from lib.tool.raptorx.Config import CONFIGS
from lib.tool.raptorx.Dataset import SeqDataset
from lib.tool.raptorx.models.MainModel import MainModel
from lib.tool.raptorx.utils.Utils import outputs_to_pdb

import logging

FORMAT = "[%(filename)s:%(lineno)s %(funcName)s]: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__MYPATH__ = os.path.split(os.path.realpath(__file__))[0]


class Attn4StrucPred:
    def __init__(self, configs, param, device, plm_models, n_cycle, out_dir, outfile_tag):
        super().__init__()

        self.device = device
        self.n_cycle = n_cycle
        self.out_dir = out_dir
        self.outfile_tag = outfile_tag
        self.pred_keys = configs["PREDICTION"]["pred_keys"]
        configs["EMBEDDING_MODULE"]["plm_config"]["plm_models"] = plm_models
        configs["EMBEDDING_MODULE"]["plm_config"]["model_config"][
            "device"
        ] = device

        # model
        self.model = MainModel(configs).to(self.device)
        
        # load params
        state_dict = torch.load(param, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x, recycling, recycle_data, *args, **kwargs):
        x["recycling"] = recycling
        x["recycle_data"] = recycle_data

        preds, recycle_data = self.model(x, *args, **kwargs)

        return preds, recycle_data

    def move_to_device(self, batch_data):
        for k in batch_data[0]:
            if isinstance(batch_data[0][k], torch.Tensor):
                batch_data[0][k] = batch_data[0][k].to(self.device)

        return batch_data

    def pred_step(self, batch_data, batch_idx):
        feature, sample_info = self.move_to_device(batch_data)

        start_time = time.time()

        target = sample_info["target"][0]

        recycle_data = None
        for i in range(self.n_cycle):
            preds, recycle_data = self.forward(
                feature, i < self.n_cycle, recycle_data
            )

        # save pdb
        outputs_to_pdb(
            preds, sample_info, self.pred_keys, self.out_dir, self.outfile_tag
        )

        print(f"running time: {target}", time.time() - start_time)

    def pred(self, data_loader):
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                self.pred_step(sample, i)

        print("total running time:", time.time() - start_time)


def prediction(fasta_path, param, plm_param_dir, n_cycle, out_dir, outfile_tag):
    device = ""
    
    plm_models = []
    plm_models += ["ProtTrans"]
    os.environ["ProtTrans_param"] = plm_param_dir + "prot_t5_xl_uniref50"

    data_set = SeqDataset(fasta_path, plm_models)
    data_loader = DataLoader(data_set, pin_memory=False, num_workers=4)

    mAttn4StrucPred = Attn4StrucPred(CONFIGS, param, device, plm_models, n_cycle, out_dir, outfile_tag)
    mAttn4StrucPred.pred(data_loader)
