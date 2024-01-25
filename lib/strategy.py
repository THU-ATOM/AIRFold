from typing import Any, Dict
from lib.base import BasePathTree


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
            p_ = p_ / (method_[:5] + parse_strgy(str_dict[method_]))
            path_l.append(p_ / f"{self.id}.a3m")
        # return the list of
        return path_l

    @property
    def final_fasta(self):
        if len(self.strategy_list) > 0:
            return self.strategy_list[-1]
        else:
            return None