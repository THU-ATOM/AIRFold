import argparse

import pickle as pkl


def main(
    template_feature_path,
    selected_template_feature_path,
    remain_num=4,
):
    with open(template_feature_path, "rb") as fd:
        tplt_feature = pkl.load(fd)

    selected_feature = {k: tplt_feature[k][:remain_num] for k in tplt_feature}
    with open(selected_template_feature_path, "wb") as fd:
        pkl.dump(selected_feature, fd)
    return selected_template_feature_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_feat", type=str, required=True)
    parser.add_argument("-t", "--tgt_feat", type=str, required=True)
    parser.add_argument("-m", "--max_pool_size", type=int, default=20)
    parser.add_argument("-n", "--remain_num", type=int, default=4)
    args = parser.parse_args()
    main(
        template_feature_path=args.src_feat,
        selected_template_feature_path=args.tgt_feat,
        remain_num=args.remain_num,
    )
