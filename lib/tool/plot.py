import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects
from string import ascii_uppercase, ascii_lowercase
import numpy as np
import py3Dmol
from matplotlib import collections as mcoll


pymol_color_list = [
    "#33ff33",
    "#00ffff",
    "#ff33cc",
    "#ffff00",
    "#ff9999",
    "#e5e5e5",
    "#7f7fff",
    "#ff7f00",
    "#7fff7f",
    "#199999",
    "#ff007f",
    "#ffdd5e",
    "#8c3f99",
    "#b2b2b2",
    "#007fff",
    "#c4b200",
    "#8cb266",
    "#00bfbf",
    "#b27f7f",
    "#fcd1a5",
    "#ff7f7f",
    "#ffbfdd",
    "#7fffff",
    "#ffff7f",
    "#00ff7f",
    "#337fcc",
    "#d8337f",
    "#bfff3f",
    "#ff7fff",
    "#d8d8ff",
    "#3fffbf",
    "#b78c4c",
    "#339933",
    "#66b2b2",
    "#ba8c84",
    "#84bf00",
    "#b24c66",
    "#7f7f7f",
    "#3f3fa5",
    "#a5512b",
]

pymol_cmap = matplotlib.colors.ListedColormap(pymol_color_list)
alphabet_list = list(ascii_uppercase + ascii_lowercase)


def plot_plddt_legend(dpi=100):
    thresh = [
        "plDDT:",
        "Very low (<50)",
        "Low (60)",
        "OK (70)",
        "Confident (80)",
        "Very high (>90)",
    ]
    plt.figure(figsize=(1, 0.1), dpi=dpi)
    ########################################
    for c in ["#FFFFFF", "#FF0000", "#FFFF00", "#00FF00", "#00FFFF", "#0000FF"]:
        plt.bar(0, 0, color=c)
    plt.legend(
        thresh,
        frameon=False,
        loc="center",
        ncol=6,
        handletextpad=1,
        columnspacing=1,
        markerscale=0.5,
    )
    plt.axis(False)
    return plt


def plot_ticks(Ls):
    Ln = sum(Ls)
    L_prev = 0
    for L_i in Ls[:-1]:
        L = L_prev + L_i
        L_prev += L_i
        plt.plot([0, Ln], [L, L], color="black")
        plt.plot([L, L], [0, Ln], color="black")
    ticks = np.cumsum([0] + Ls)
    ticks = (ticks[1:] + ticks[:-1]) / 2
    plt.yticks(ticks, alphabet_list[: len(ticks)])


def plot_confidence(plddt, pae=None, Ls=None, dpi=100):
    use_ptm = False if pae is None else True
    if use_ptm:
        plt.figure(figsize=(10, 3), dpi=dpi)
        plt.subplot(1, 2, 1)
    else:
        plt.figure(figsize=(5, 3), dpi=dpi)
    plt.title("Predicted lDDT")
    plt.plot(plddt)
    if Ls is not None:
        L_prev = 0
        for L_i in Ls[:-1]:
            L = L_prev + L_i
            L_prev += L_i
            plt.plot([L, L], [0, 100], color="black")
    plt.ylim(0, 100)
    plt.ylabel("plDDT")
    plt.xlabel("position")
    if use_ptm:
        plt.subplot(1, 2, 2)
        plt.title("Predicted Aligned Error")
        Ln = pae.shape[0]
        plt.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, Ln, Ln, 0))
        if Ls is not None and len(Ls) > 1:
            plot_ticks(Ls)
        plt.colorbar()
        plt.xlabel("Scored residue")
        plt.ylabel("Aligned residue")
    return plt


def plot_msas(
    msas,
    ori_seq=None,
    sort_by_seqid=True,
    deduplicate=True,
    dpi=100,
    return_plt=True,
):
    """
    plot the msas
    """
    if ori_seq is None:
        ori_seq = msas[0][0]
    seqs = ori_seq.replace("/", "").split(":")
    seqs_dash = ori_seq.replace(":", "").split("/")

    Ln = np.cumsum(np.append(0, [len(seq) for seq in seqs]))
    Ln_dash = np.cumsum(np.append(0, [len(seq) for seq in seqs_dash]))
    Nn, lines = [], []
    for msa in msas:
        msa_ = set(msa) if deduplicate else msa
        if len(msa_) > 0:
            Nn.append(len(msa_))
            msa_ = np.asarray([list(seq) for seq in msa_])
            gap_ = msa_ != "-"
            qid_ = msa_ == np.array(list("".join(seqs)))
            print(gap_.shape, msa_.shape)
            gapid = np.stack(
                [gap_[:, Ln[i] : Ln[i + 1]].max(-1) for i in range(len(seqs))],
                -1,
            )
            seqid = np.stack(
                [qid_[:, Ln[i] : Ln[i + 1]].mean(-1) for i in range(len(seqs))],
                -1,
            ).sum(-1) / (gapid.sum(-1) + 1e-8)
            non_gaps = gap_.astype(np.float)
            non_gaps[non_gaps == 0] = np.nan
            if sort_by_seqid:
                lines.append(non_gaps[seqid.argsort()] * seqid[seqid.argsort(), None])
            else:
                lines.append(non_gaps[::-1] * seqid[::-1, None])

    Nn = np.cumsum(np.append(0, Nn))
    lines = np.concatenate(lines, 0)

    if return_plt:
        plt.figure(figsize=(8, 5), dpi=dpi)
        plt.title("Sequence coverage")
    plt.imshow(
        lines,
        interpolation="nearest",
        aspect="auto",
        cmap="rainbow_r",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=(0, lines.shape[1], 0, lines.shape[0]),
    )
    for i in Ln[1:-1]:
        plt.plot([i, i], [0, lines.shape[0]], color="black")
    for i in Ln_dash[1:-1]:
        plt.plot([i, i], [0, lines.shape[0]], "--", color="black")
    for j in Nn[1:-1]:
        plt.plot([0, lines.shape[1]], [j, j], color="black")

    plt.plot((np.isnan(lines) == False).sum(0), color="black")
    plt.xlim(0, lines.shape[1])
    plt.ylim(0, lines.shape[0])
    plt.colorbar(label="Sequence identity to query")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    if return_plt:
        return plt


def read_pdb_renum(pdb_filename, Ls=None):
    if Ls is not None:
        L_init = 0
        new_chain = {}
        for L, c in zip(Ls, alphabet_list):
            new_chain.update({i: c for i in range(L_init, L_init + L)})
            L_init += L

    n, pdb_out = 1, []
    resnum_, chain_ = 1, "A"
    for line in open(pdb_filename, "r"):
        if line[:4] == "ATOM":
            chain = line[21:22]
            resnum = int(line[22 : 22 + 5])
            if resnum != resnum_ or chain != chain_:
                resnum_, chain_ = resnum, chain
                n += 1
            if Ls is None:
                pdb_out.append("%s%4i%s" % (line[:22], n, line[26:]))
            else:
                pdb_out.append(
                    "%s%s%4i%s" % (line[:21], new_chain[n - 1], n, line[26:])
                )
    return "".join(pdb_out)


def show_pdb(
    pred_output_path,
    show_sidechains=False,
    show_mainchains=False,
    color="lDDT",
    chains=None,
    Ls=None,
    vmin=50,
    vmax=90,
    color_HP=False,
    size=(800, 480),
):

    if chains is None:
        chains = 1 if Ls is None else len(Ls)

    view = py3Dmol.view(
        js="https://3dmol.org/build/3Dmol.js", width=size[0], height=size[1]
    )
    view.addModel(read_pdb_renum(pred_output_path, Ls), "pdb")
    if color == "lDDT":
        view.setStyle(
            {
                "cartoon": {
                    "colorscheme": {
                        "prop": "b",
                        "gradient": "roygb",
                        "min": vmin,
                        "max": vmax,
                    }
                }
            }
        )
    elif color == "rainbow":
        view.setStyle({"cartoon": {"color": "spectrum"}})
    elif color == "chain":
        for n, chain, color in zip(range(chains), alphabet_list, pymol_color_list):
            view.setStyle({"chain": chain}, {"cartoon": {"color": color}})
    if show_sidechains:
        BB = ["C", "O", "N"]
        HP = [
            "ALA",
            "GLY",
            "VAL",
            "ILE",
            "LEU",
            "PHE",
            "MET",
            "PRO",
            "TRP",
            "CYS",
            "TYR",
        ]
        if color_HP:
            view.addStyle(
                {"and": [{"resn": HP}, {"atom": BB, "invert": True}]},
                {"stick": {"colorscheme": "yellowCarbon", "radius": 0.3}},
            )
            view.addStyle(
                {
                    "and": [
                        {"resn": HP, "invert": True},
                        {"atom": BB, "invert": True},
                    ]
                },
                {"stick": {"colorscheme": "whiteCarbon", "radius": 0.3}},
            )
            view.addStyle(
                {"and": [{"resn": "GLY"}, {"atom": "CA"}]},
                {"sphere": {"colorscheme": "yellowCarbon", "radius": 0.3}},
            )
            view.addStyle(
                {
                    "and": [
                        {"resn": "PRO"},
                        {"atom": ["C", "O"], "invert": True},
                    ]
                },
                {"stick": {"colorscheme": "yellowCarbon", "radius": 0.3}},
            )
        else:
            view.addStyle(
                {
                    "and": [
                        {"resn": ["GLY", "PRO"], "invert": True},
                        {"atom": BB, "invert": True},
                    ]
                },
                {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
            )
            view.addStyle(
                {"and": [{"resn": "GLY"}, {"atom": "CA"}]},
                {"sphere": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
            )
            view.addStyle(
                {
                    "and": [
                        {"resn": "PRO"},
                        {"atom": ["C", "O"], "invert": True},
                    ]
                },
                {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
            )
    if show_mainchains:
        BB = ["C", "O", "N", "CA"]
        view.addStyle(
            {"atom": BB},
            {"stick": {"colorscheme": f"WhiteCarbon", "radius": 0.3}},
        )
    view.zoomTo()
    return view


def plot_plddts(plddts, Ls=None, dpi=100, fig=True):
    if fig:
        plt.figure(figsize=(8, 5), dpi=100)
    plt.title("Predicted lDDT per position")
    for name, plddt in plddts.items():
        plt.plot(plddt, label=f"{name}")
    if Ls is not None:
        L_prev = 0
        for L_i in Ls[:-1]:
            L = L_prev + L_i
            L_prev += L_i
            plt.plot([L, L], [0, 100], color="black")
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("Predicted lDDT")
    plt.xlabel("Positions")
    return plt


def plot_lddts(lddts, Ls=None, dpi=100, fig=True):
    if fig:
        plt.figure(figsize=(8, 5), dpi=100)
    plt.title("LDDT per position")
    for name, lddt in lddts.items():
        plt.plot(lddt, label=f"{name}")
    if Ls is not None:
        L_prev = 0
        for L_i in Ls[:-1]:
            L = L_prev + L_i
            L_prev += L_i
            plt.plot([L, L], [0, 100], color="black")
    plt.legend()
    plt.ylim(0, 100)
    plt.ylabel("LDDT")
    plt.xlabel("Positions")
    return plt


def plot_paes(paes, Ls=None, dpi=100, fig=True):
    num_models = len(paes)
    if fig:
        plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
    for n, pae in enumerate(paes):
        plt.subplot(1, num_models, n + 1)
        plt.title(f"rank_{n+1}")
        Ln = pae.shape[0]
        plt.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, Ln, Ln, 0))
        if Ls is not None and len(Ls) > 1:
            plot_ticks(Ls)
        plt.colorbar()
    return plt


def plot_adjs(adjs, Ls=None, dpi=100, fig=True):
    num_models = len(adjs)
    if fig:
        plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
    for n, adj in enumerate(adjs):
        plt.subplot(1, num_models, n + 1)
        plt.title(f"rank_{n+1}")
        Ln = adj.shape[0]
        plt.imshow(adj, cmap="binary", vmin=0, vmax=1, extent=(0, Ln, Ln, 0))
        if Ls is not None and len(Ls) > 1:
            plot_ticks(Ls)
        plt.colorbar()
    return plt


def plot_dists(dists, Ls=None, dpi=100, fig=True):
    num_models = len(dists)
    if fig:
        plt.figure(figsize=(3 * num_models, 2), dpi=dpi)
    for n, dist in enumerate(dists):
        plt.subplot(1, num_models, n + 1)
        plt.title(f"rank_{n+1}")
        Ln = dist.shape[0]
        plt.imshow(dist, extent=(0, Ln, Ln, 0))
        if Ls is not None and len(Ls) > 1:
            plot_ticks(Ls)
        plt.colorbar()
    return plt


##########################################################################
##########################################################################


def kabsch(a, b, weights=None, return_v=False):
    a = np.asarray(a)
    b = np.asarray(b)
    if weights is None:
        weights = np.ones(len(b))
    else:
        weights = np.asarray(weights)
    B = np.einsum("ji,jk->ik", weights[:, None] * a, b)
    u, s, vh = np.linalg.svd(B)
    if np.linalg.det(u @ vh) < 0:
        u[:, -1] = -u[:, -1]
    if return_v:
        return u
    else:
        return u @ vh


def plot_pseudo_3D(
    xyz,
    c=None,
    ax=None,
    chainbreak=5,
    cmap="gist_rainbow",
    line_w=2.0,
    cmin=None,
    cmax=None,
    zmin=None,
    zmax=None,
):
    def rescale(a, amin=None, amax=None):
        a = np.copy(a)
        if amin is None:
            amin = a.min()
        if amax is None:
            amax = a.max()
        a[a < amin] = amin
        a[a > amax] = amax
        return (a - amin) / (amax - amin)

    # make segments
    xyz = np.asarray(xyz)
    seg = np.concatenate([xyz[:-1, None, :], xyz[1:, None, :]], axis=-2)
    seg_xy = seg[..., :2]
    seg_z = seg[..., 2].mean(-1)
    ord = seg_z.argsort()

    # set colors
    if c is None:
        c = np.arange(len(seg))[::-1]
    else:
        c = (c[1:] + c[:-1]) / 2
    c = rescale(c, cmin, cmax)

    if isinstance(cmap, str):
        if cmap == "gist_rainbow":
            c *= 0.75
        colors = matplotlib.cm.get_cmap(cmap)(c)
    else:
        colors = cmap(c)

    if chainbreak is not None:
        dist = np.linalg.norm(xyz[:-1] - xyz[1:], axis=-1)
        colors[..., 3] = (dist < chainbreak).astype(np.float)

    # add shade/tint based on z-dimension
    z = rescale(seg_z, zmin, zmax)[:, None]
    tint, shade = z / 3, (z + 2) / 3
    colors[:, :3] = colors[:, :3] + (1 - colors[:, :3]) * tint
    colors[:, :3] = colors[:, :3] * shade

    set_lim = False
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_figwidth(5)
        fig.set_figheight(5)
        set_lim = True
    else:
        fig = ax.get_figure()
        if ax.get_xlim() == (0, 1):
            set_lim = True

    if set_lim:
        xy_min = xyz[:, :2].min() - line_w
        xy_max = xyz[:, :2].max() + line_w
        ax.set_xlim(xy_min, xy_max)
        ax.set_ylim(xy_min, xy_max)

    ax.set_aspect("equal")

    # determine linewidths
    width = fig.bbox_inches.width * ax.get_position().width
    linewidths = line_w * 72 * width / np.diff(ax.get_xlim())

    lines = mcoll.LineCollection(
        seg_xy[ord],
        colors=colors[ord],
        linewidths=linewidths,
        path_effects=[matplotlib.patheffects.Stroke(capstyle="round")],
    )

    return ax.add_collection(lines)


def add_text(text, ax):
    return plt.text(
        0.5,
        1.01,
        text,
        horizontalalignment="center",
        verticalalignment="bottom",
        transform=ax.transAxes,
    )


def plot_protein(
    protein=None,
    pos=None,
    plddt=None,
    Ls=None,
    dpi=100,
    best_view=True,
    line_w=2.0,
):

    if protein is not None:
        pos = np.asarray(protein.atom_positions[:, 1, :])
        plddt = np.asarray(protein.b_factors[:, 0])

    # get best view
    if best_view:
        if plddt is not None:
            weights = plddt / 100
            pos = pos - (pos * weights[:, None]).sum(0, keepdims=True) / weights.sum()
            pos = pos @ kabsch(pos, pos, weights, return_v=True)
        else:
            pos = pos - pos.mean(0, keepdims=True)
            pos = pos @ kabsch(pos, pos, return_v=True)

    if plddt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ax = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_figwidth(3)
        fig.set_figheight(3)
        ax = [ax1]

    fig.set_dpi(dpi)
    fig.subplots_adjust(top=0.9, bottom=0.1, right=1, left=0, hspace=0, wspace=0)

    xy_min = pos[..., :2].min() - line_w
    xy_max = pos[..., :2].max() + line_w
    for a in ax:
        a.set_xlim(xy_min, xy_max)
        a.set_ylim(xy_min, xy_max)
        a.axis(False)

    if Ls is None or len(Ls) == 1:
        # color N->C
        c = np.arange(len(pos))[::-1]
        plot_pseudo_3D(pos, line_w=line_w, ax=ax1)
        add_text("colored by N→C", ax1)
    else:
        # color by chain
        c = np.concatenate([[n] * L for n, L in enumerate(Ls)])
        if len(Ls) > 40:
            plot_pseudo_3D(pos, c=c, line_w=line_w, ax=ax1)
        else:
            plot_pseudo_3D(
                pos,
                c=c,
                cmap=pymol_cmap,
                cmin=0,
                cmax=39,
                line_w=line_w,
                ax=ax1,
            )
        add_text("colored by chain", ax1)

    if plddt is not None:
        # color by pLDDT
        plot_pseudo_3D(pos, c=plddt, cmin=50, cmax=90, line_w=line_w, ax=ax2)
        add_text("colored by pLDDT", ax2)

    return fig


def plot_protein_lddts(
    protein=None,
    pos=None,
    lddt=None,
    plddt=None,
    dpi=100,
    best_view=True,
    line_w=2.0,
):

    if protein is not None:
        pos = np.asarray(protein.atom_positions[:, 1, :])
        plddt = np.asarray(protein.b_factors[:, 0])

    # get best view
    if best_view:
        if plddt is not None:
            weights = plddt / 100
            pos = pos - (pos * weights[:, None]).sum(0, keepdims=True) / weights.sum()
            pos = pos @ kabsch(pos, pos, weights, return_v=True)
        else:
            pos = pos - pos.mean(0, keepdims=True)
            pos = pos @ kabsch(pos, pos, return_v=True)

    if plddt is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_figwidth(6)
        fig.set_figheight(3)
        ax = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_figwidth(3)
        fig.set_figheight(3)
        ax = [ax1]

    fig.set_dpi(dpi)
    fig.subplots_adjust(top=0.9, bottom=0.1, right=1, left=0, hspace=0, wspace=0)

    xy_min = pos[..., :2].min() - line_w
    xy_max = pos[..., :2].max() + line_w
    for a in ax:
        a.set_xlim(xy_min, xy_max)
        a.set_ylim(xy_min, xy_max)
        a.axis(False)

    if lddt is None:
        # color N->C
        c = np.arange(len(pos))[::-1]
        plot_pseudo_3D(pos, line_w=line_w, ax=ax1)
        add_text("colored by N→C", ax1)
    else:
        # color by lddt
        lddt = np.asarray(lddt)
        plot_pseudo_3D(pos, c=lddt, cmin=50, cmax=90, line_w=line_w, ax=ax1)
        add_text("colored by LDDT", ax1)

    if plddt is not None:
        # color by pLDDT
        plot_pseudo_3D(pos, c=plddt, cmin=50, cmax=90, line_w=line_w, ax=ax2)
        add_text("colored by pLDDT", ax2)

    return fig
