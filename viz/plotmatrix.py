import os
import json
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt


CfgLike = Union[str, dict]


def _load_cfg(cfg: CfgLike) -> dict:
    if isinstance(cfg, str):
        with open(cfg, "r") as f:
            return json.load(f)
    return dict(cfg)


def _to_bins(start_bp: int, end_bp: int, res: int) -> Tuple[int, int]:
    start_bin = int(start_bp // res)
    end_bin = int(np.ceil(end_bp / float(res)))
    return start_bin, end_bin


def _unit_factor(unit: str) -> Tuple[float, str]:
    """Return (factor, label) for genomic axis scaling.
    unit in {'bp','kb','Mb'}; default to 'bp' for unknown inputs.
    factor multiplies raw bp values to axis units.
    """
    u = (unit or 'bp').lower()
    if u in ('mb', 'm'):  # megabase
        return 1e-6, 'Mb'
    if u in ('kb', 'k'):  # kilobase
        return 1e-3, 'kb'
    return 1.0, 'bp'


def _agg(vals: np.ndarray, axis: int, how: str) -> np.ndarray:
    how = (how or "mean").lower()
    if how == "median":
        return np.median(vals, axis=axis)
    return np.mean(vals, axis=axis)


def _build_matrix_impute(
    f5: h5py.File,
    coords: np.ndarray,
    start_bin: int,
    end_bin: int,
    group: Optional[Union[int, Sequence[int]]],
    how: str = "mean",
    symmetrize: bool = True,
) -> Tuple[np.ndarray, str]:
    n = end_bin - start_bin
    coords_mask = (
        (coords[:, 0] >= start_bin)
        & (coords[:, 0] < end_bin)
        & (coords[:, 1] >= start_bin)
        & (coords[:, 1] < end_bin)
    )
    coords_sub = coords[coords_mask] - start_bin

    # collect available cell_* keys
    cell_keys = [k for k in f5.keys() if k.startswith("cell_")]
    avail = sorted(int(k.split("_")[1]) for k in cell_keys)

    # decide which cells to aggregate
    if group is None:
        # mean over all available
        mats = []
        for ci in avail:
            v = np.asarray(f5[f"cell_{ci}"][:], dtype=np.float32)
            mats.append(v[coords_mask])
        vals_sub = _agg(np.stack(mats, axis=0), axis=0, how=how)
        label = f"Hi-C (impute {how} across {len(avail)} cells)"
    elif np.isscalar(group):
        ci = int(group)
        if ci not in avail:
            raise IndexError(f"Hi-C cell index {ci} not in available {avail[:5]} ...")
        vals = np.asarray(f5[f"cell_{ci}"][:], dtype=np.float32)
        vals_sub = vals[coords_mask]
        label = f"Hi-C (impute cell {ci})"
    else:
        sel = np.unique(np.asarray(list(group), dtype=int))
        sel = [ci for ci in sel if ci in avail]
        if len(sel) == 0:
            raise ValueError("Selected Hi-C cell indices not found in HDF5.")
        mats = []
        for ci in sel:
            v = np.asarray(f5[f"cell_{ci}"][:], dtype=np.float32)
            mats.append(v[coords_mask])
        vals_sub = _agg(np.stack(mats, axis=0), axis=0, how=how)
        label = f"Hi-C (impute {how} of {len(sel)} cells)"

    # place values into dense matrix
    mat = np.zeros((n, n), dtype=np.float32)
    mat[coords_sub[:, 0], coords_sub[:, 1]] = vals_sub
    if symmetrize:
        mat[coords_sub[:, 1], coords_sub[:, 0]] = vals_sub
    return mat, label


def _build_matrix_raw(
    origin_sparse: np.ndarray,
    start_bin: int,
    end_bin: int,
    group: Optional[Union[int, Sequence[int]]],
    how: str = "mean",
    symmetrize: bool = True,
) -> Tuple[np.ndarray, str]:
    n_cells = len(origin_sparse)

    def sub_for_cell(ci: int) -> np.ndarray:
        if not (0 <= ci < n_cells):
            raise IndexError(f"Hi-C cell index {ci} out of range [0,{n_cells-1}]")
        A = origin_sparse[ci][start_bin:end_bin, start_bin:end_bin]
        arr = A.toarray().astype(np.float32)
        if symmetrize:
            arr = (arr + arr.T) - np.diag(np.diag(arr))
        return arr

    if group is None:
        mats = [sub_for_cell(ci) for ci in range(n_cells)]
        mat = _agg(np.stack(mats, axis=0), axis=0, how=how)
        label = f"Hi-C (raw {how} across {n_cells} cells)"
    elif np.isscalar(group):
        ci = int(group)
        mat = sub_for_cell(ci)
        label = f"Hi-C (raw cell {ci})"
    else:
        sel = np.unique(np.asarray(list(group), dtype=int))
        sel_valid = [ci for ci in sel if 0 <= ci < n_cells]
        if len(sel_valid) == 0:
            raise ValueError(f"Selected Hi-C cell indices not in range [0,{n_cells-1}].")
        mats = [sub_for_cell(ci) for ci in sel_valid]
        mat = _agg(np.stack(mats, axis=0), axis=0, how=how)
        label = f"Hi-C (raw {how} of {len(sel_valid)} cells)"
    return mat, label


def _load_imputed_atac(temp_dir: str, chrom: str, prefer: str = "auto") -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Load per-cell x per-bin ATAC arrays from temp, if available.
    Search order depends on `prefer`:
      - 'prob' -> imputed_atac_{chrom}_prob.npy (probability in [0,1])
      - 'z'    -> imputed_atac_{chrom}.npy     (z-score)
      - 'raw'  -> imputed_atac_{chrom}_raw.npy (approx raw)
      - 'auto' -> try 'prob' then 'z' then 'raw'

    Returns (array or None, tag)
    """
    order = []
    p = prefer.lower()
    if p == "prob":
        order = ["prob"]
    elif p == "z":
        order = ["z"]
    elif p == "raw":
        order = ["raw"]
    else:
        order = ["prob", "z", "raw"]
    for tag in order:
        if tag == "prob":
            path = os.path.join(temp_dir, f"imputed_atac_{chrom}_prob.npy")
        elif tag == "z":
            path = os.path.join(temp_dir, f"imputed_atac_{chrom}.npy")
        else:
            path = os.path.join(temp_dir, f"imputed_atac_{chrom}_raw.npy")
        if os.path.exists(path):
            try:
                arr = np.load(path)
                return arr, tag
            except Exception:
                pass
    return None, None


def _load_raw_atac_from_h5(cfg: dict, chrom: str, start_bin: int, end_bin: int) -> Tuple[np.ndarray, List[int]]:
    """Stream-load raw coassay (e.g., ATAC) per-cell vectors for the chrom and slice region.
    Returns array of shape [n_cells, bins_in_region] and a sorted list of valid cell indices.
    """
    data_dir = cfg.get("data_dir", "data")
    signal_names = cfg.get("coassay_signal", [])
    if not signal_names:
        raise RuntimeError("coassay_signal is empty; cannot load raw coassay from HDF5")
    import h5py
    h5_path = os.path.join(data_dir, 'sc_signal.hdf5')
    with h5py.File(h5_path, 'r') as h5:
        if signal_names[0] not in h5:
            raise RuntimeError(f"group {signal_names[0]} not found in sc_signal.hdf5")
        g = h5[signal_names[0]]
        try:
            chrom_arr = g['bin']['chrom'].asstr()[...]
        except AttributeError:
            chrom_arr = np.array([c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in g['bin']['chrom'][...]])
        mask = (chrom_arr == chrom)
        # Collect numeric keys only
        cell_keys = sorted([int(k) for k in g.keys() if k.isdigit()])
        region_len = end_bin - start_bin
        out = np.zeros((len(cell_keys), region_len), dtype=np.float32)
        for idx, ci in enumerate(cell_keys):
            v_full = np.array(g[str(ci)])[mask]
            out[idx, :] = v_full[start_bin:end_bin]
    return out, cell_keys


def _build_coassay_profile(
    cfg: dict,
    chrom: str,
    start_bin: int,
    end_bin: int,
    group: Optional[Union[int, Sequence[int]]],
    prefer: str = "auto",
    how: str = "mean",
    norm: str = "minmax",
) -> Tuple[np.ndarray, str]:
    """Aggregate a coassay (ATAC) 1D profile for a group within [start_bin,end_bin).
    Prefer imputed arrays if present; fall back to raw HDF5.
    norm: 'none' | 'minmax' | 'zscore'
    """
    temp_dir = cfg.get("temp_dir", "temp")
    arr, tag = _load_imputed_atac(temp_dir, chrom, prefer=prefer)
    label_tag = tag or "raw"
    if arr is not None:
        # arr: [n_cells, n_bins_on_chrom]
        region = arr[:, start_bin:end_bin].astype(np.float32)
        avail = list(range(region.shape[0]))
    else:
        # raw HDF5
        region, avail = _load_raw_atac_from_h5(cfg, chrom, start_bin, end_bin)

    if group is None:
        prof = _agg(region, axis=0, how=how)
        glabel = f"Coassay ({label_tag} {how} all)"
    elif np.isscalar(group):
        ci = int(group)
        if ci not in avail:
            raise IndexError(f"Coassay cell index {ci} not in available {avail[:5]} ...")
        prof = region[ci if arr is not None else avail.index(ci)]
        glabel = f"Coassay ({label_tag} cell {ci})"
    else:
        sel = np.unique(np.asarray(list(group), dtype=int))
        # Map selection to row indices
        if arr is not None:
            valid = [ci for ci in sel if 0 <= ci < region.shape[0]]
            if len(valid) == 0:
                raise ValueError("Selected coassay cell indices are empty for imputed arrays")
            prof = _agg(region[valid, :], axis=0, how=how)
        else:
            valid = [ci for ci in sel if ci in avail]
            if len(valid) == 0:
                raise ValueError("Selected coassay cell indices not found in HDF5")
            rows = [avail.index(ci) for ci in valid]
            prof = _agg(region[rows, :], axis=0, how=how)
        glabel = f"Coassay ({label_tag} {how} of {len(valid)})"

    # normalize for view if requested
    n = norm.lower() if isinstance(norm, str) else "minmax"
    if n == "minmax":
        lo, hi = float(np.min(prof)), float(np.max(prof))
        span = hi - lo if hi > lo else 1.0
        prof = (prof - lo) / span
    elif n == "zscore":
        mu, sd = float(np.mean(prof)), float(np.std(prof) + 1e-6)
        prof = (prof - mu) / sd
    # else 'none'
    return prof.astype(np.float32), glabel


def plot_hic_region_grid(
    cfg: CfgLike,
    chrom: str,
    start_bp: int,
    end_bp: int,
    kind: str = "impute",
    *groups: Optional[Union[int, Sequence[int]]],
    group_names: Optional[Sequence[str]] = None,
    nbr_k: int = 0,
    how: str = "mean",
    symmetrize: bool = True,
    cmap: str = "magma",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    sharex: bool = True,
    sharey: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    colorbar: bool = True,
    percentile_clip: Optional[float] = 99.0,
    # Coassay track options
    show_coassay: bool = True,
    coassay_prefer: str = "auto",  # 'auto' | 'prob' | 'z' | 'raw'
    coassay_norm: str = "minmax",
    coassay_ymin: Optional[float] = None,
    coassay_ymax: Optional[float] = None,
    # Colorbar axis width (as a fraction of a data axis)
    cbar_width: float = 0.05,
    # Genomic axis unit
    unit: str = 'bp',  # 'bp' | 'kb' | 'Mb'
) -> Tuple[plt.Figure, np.ndarray, List[np.ndarray]]:
    """Plot Hi-C submatrices for multiple groups side by side.

    Parameters
    ----------
    cfg : str | dict
        Path to config.JSON or loaded dict.
    chrom : str
        Chromosome name, e.g., 'chr1'.
    start_bp, end_bp : int
        Genomic range in basepairs (half-open [start, end)).
    kind : {'impute','raw'}
        Use imputed HDF5 or raw sparse adjacencies.
    *groups : sequence
        Variable number of group specs. Each can be:
        - None: mean across all cells
        - int: single cell index
        - sequence of int: aggregate subset
    group_names : list[str], optional
        Names shown above each column. If None, auto-generated.
    nbr_k : int
        Neighbor k used in impute file naming (0 => no neighbors).
    how : {'mean','median'}
        Aggregation when group is None or a list.
    symmetrize : bool
        Symmetrize matrices.
    cmap : str
        Matplotlib colormap.
    vmin, vmax : float, optional
        Color scaling. If None, computed from data (optionally percentile clipped).
    sharex, sharey : bool
        Share axes among subplots.
    figsize : (w, h), optional
        Figure size. If None, auto scales with number of columns.
    colorbar : bool
        Whether to draw a single colorbar to the right.
    percentile_clip : float, optional
        If set, clip vmax to this percentile across all values (robust).

    Returns
    -------
    fig, axes, mats : Figure, Axes array, list of ndarray
        Matplotlib figure/axes and the list of matrices per column.
    """
    cfg = _load_cfg(cfg)
    temp_dir = cfg["temp_dir"]
    res = int(cfg["resolution"])
    start_bin, end_bin = _to_bins(start_bp, end_bp, res)
    fac, unit_lbl = _unit_factor(unit)

    if len(groups) == 0:
        groups = (None,)  # single column, mean across all cells

    # build matrices per group
    mats: List[np.ndarray] = []
    labels: List[str] = []

    kind_l = kind.lower()
    if kind_l in ("impute", "imputed"):
        embedding_name = cfg["embedding_name"]
        hic_path = os.path.join(temp_dir, f"{chrom}_{embedding_name}_nbr_{nbr_k}_impute.hdf5")
        if not os.path.exists(hic_path):
            raise FileNotFoundError(
                f"Not found: {hic_path}. Run impute_no_nbr()/impute_with_nbr() first."
            )
        with h5py.File(hic_path, "r") as f5:
            coords = np.asarray(f5["coordinates"], dtype=np.int64)
            for g in groups:
                mat, label = _build_matrix_impute(
                    f5, coords, start_bin, end_bin, g, how=how, symmetrize=symmetrize
                )
                mats.append(mat)
                labels.append(label)
    elif kind_l == "raw":
        raw_path = os.path.join(temp_dir, "raw", f"{chrom}_sparse_adj.npy")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(
                f"Not found: {raw_path}. Generate raw sparse matrices via create_matrix()."
            )
        origin_sparse = np.load(raw_path, allow_pickle=True)
        for g in groups:
            mat, label = _build_matrix_raw(
                origin_sparse, start_bin, end_bin, g, how=how, symmetrize=symmetrize
            )
            mats.append(mat)
            labels.append(label)
    else:
        raise ValueError("kind must be 'impute' or 'raw'")

    # compute color range
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        all_vals = np.concatenate([m.ravel() for m in mats])
        if percentile_clip is not None:
            vmax = float(np.percentile(all_vals, float(percentile_clip)))
        else:
            vmax = float(all_vals.max(initial=1.0))
        if vmax <= 0:
            vmax = 1.0

    ncols = len(mats)
    if figsize is None:
        # Height accounts for optional coassay track
        base_h = 4.0 if not show_coassay else 5.5
        figsize = (4.0 * ncols + (0.6 if colorbar else 0.0), base_h)
    # Build a grid for data only; colorbar will be attached outside with ax=...
    nrows = 2 if show_coassay else 1
    width_ratios = [1.0] * ncols
    height_ratios = [1.0, 3.0] if show_coassay else None
    # Avoid sharing y between signal row and image row; only share x
    sharey_grid = False if show_coassay else sharey
    fig, axes = plt.subplots(
        nrows, ncols,
        sharex=sharex, sharey=sharey_grid,
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"width_ratios": width_ratios, "height_ratios": height_ratios},
        squeeze=False,
    )

    # override labels if provided
    if group_names is not None and len(group_names) == ncols:
        titles = list(group_names)
    else:
        titles = labels

    # Optionally build coassay track per column
    if show_coassay:
        # Compute profiles first
        profiles: List[np.ndarray] = []
        plabels: List[str] = []
        for g in groups:
            prof, plab = _build_coassay_profile(
                cfg, chrom, start_bin, end_bin, g, prefer=coassay_prefer, how=how, norm=coassay_norm
            )
            profiles.append(prof)
            plabels.append(plab)
    else:
        profiles = []
        plabels = []

    # plotting per column
    for j in range(ncols):
        ax_img = axes[1, j] if show_coassay else axes[0, j]
        mat = mats[j]
        title = titles[j]
        extent = [start_bp * fac, end_bp * fac, start_bp * fac, end_bp * fac]
        im = ax_img.imshow(
            mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower",
            interpolation="nearest", aspect="auto", extent=extent
        )
        ax_img.set_title(title)
        ax_img.set_xlabel(f"Genomic position ({unit_lbl})")
        if j == 0:
            ax_img.set_ylabel(f"Genomic position ({unit_lbl})")
        else:
            ax_img.set_ylabel("")
        # Ensure genomic coordinate limits (in chosen unit)
        ax_img.set_xlim(start_bp * fac, end_bp * fac)
        ax_img.set_ylim(start_bp * fac, end_bp * fac)

        if show_coassay:
            ax_sig = axes[0, j]
            n_bins = end_bin - start_bin
            x_bp = start_bp + np.arange(n_bins) * res + res / 2.0
            ax_sig.plot(x_bp * fac, profiles[j], lw=1.2, color="tab:purple")
            ax_sig.set_xlim(start_bp * fac, end_bp * fac)
            ax_sig.set_ylabel("Coassay")
            ax_sig.set_title(title)
            # Hide x tick labels on the signal row (shared x)
            ax_sig.tick_params(labelbottom=False)
            # Set y-limits to profile range with a small margin
            p = profiles[j]
            pmin = float(np.min(p))
            pmax = float(np.max(p))
            # default limits
            if coassay_norm.lower() == "minmax":
                y0, y1 = -0.05, 1.05
            else:
                span = pmax - pmin
                if span <= 1e-12:
                    y0, y1 = pmin - 0.5, pmax + 0.5
                else:
                    y0, y1 = pmin - 0.05 * span, pmax + 0.05 * span
            # user overrides
            if coassay_ymin is not None:
                y0 = float(coassay_ymin)
            if coassay_ymax is not None:
                y1 = float(coassay_ymax)
            ax_sig.set_ylim(y0, y1)

    # attach colorbar outside the grid (no overlap with data axes)
    if colorbar:
        heatmap_axes = [axes[1, j] for j in range(ncols)] if show_coassay else [axes[0, j] for j in range(ncols)]
        fig.colorbar(im, ax=heatmap_axes, location='right', pad=0.02)

    return fig, axes, mats


# Convenience alias with a more concise name
plot_matrix_grid = plot_hic_region_grid
