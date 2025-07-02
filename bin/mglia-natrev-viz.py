#!/usr/bin/env python3
"""The nature review response for Andi's mglia paper

Second stage code, for visualization and analysis


Notes:
- drop ds 13 due to extreme outliers which distort plotting relations
    . is there a problem with the fpkm conversion or something else?

The right hand side, upper-mid groupings

- paper_3.ds_6
    . source: GSE141862
    . organism: human
    . age: 20 fetal CNS samples derived during GW9-18
    . groups: all real

- paper_6.ds_12
    . source: GSE97744
    . organism: human
    . age: fetal brain
    . groups: derived (1) and real (4)

- paper_6.ds_14
    . source: GSE99074
    . organism: human
    . age: aduly
    . groups: all real

- paper_9.ds_19
    . source: Andi's paper under review
    . organism: human
    . age: embryonic
    . groups: all real

"""

import os

import anndata as ad
import numpy as np
import scanpy as sc


def fix_filename(prefix, file_suffix):
    global out_root
    old = os.path.join(out_root, f"{prefix}{file_suffix}")
    new = os.path.join(out_root, f"{file_suffix}")
    os.rename(old, new)


if __name__ == "__main__":
    # configure data locations
    proc_root = "./mglia-natrev/procdata/"

    global out_root
    out_root = "./mglia-natrev/analysis/"
    sc.settings.figdir = out_root
    sc.settings.verbosity = 0

    print("------------------------------------------------------------------")
    print(":: loading and processing anndata...")
    print("------------------------------------------------------------------")

    # Load the processed data
    ado_file = os.path.join(proc_root, "mglia-natrev-proc.h5ad")
    ado = ad.read_h5ad(ado_file)

    # NOTE
    # One of the ds13 is still a weird outlier leading to chart perspective distortion
    ado = ado[~ado.obs["dataset"].isin(("ds13",))].copy()

    # Normalize etc
    ado.layers["counts"] = ado.X.copy()
    sc.pp.normalize_total(ado)
    sc.pp.log1p(ado)
    ado.raw = ado
    ado.X = np.nan_to_num(ado.X)

    # Calculate highly variable genes for use in pca
    # reference:
    # sc.pp.highly_variable_genes(ado, n_top_genes=2000, batch_key="sample")
    # tinker:
    # sc.pp.highly_variable_genes(ado, n_top_genes=2000, batch_key="paper")
    # sc.pp.highly_variable_genes(ado, n_top_genes=2000)
    sc.pp.highly_variable_genes(ado, n_top_genes=2000)

    # Show resulting ado
    print(ado)
    print("")
    print(ado.to_df().head(100))
    print("")

    print("------------------------------------------------------------------")
    print(":: pca and visualization...")
    print("------------------------------------------------------------------")

    # Dimensionality reduction (PCA)
    sc.tl.pca(ado)  # , mask_var="highly_variable")
    print("> completed pca Dimensionality reduction")

    # plot - show the pca components curve
    file_suffix = "mglia-natrev-pca-variance-ratio.png"
    sc.pl.pca_variance_ratio(ado, n_pcs=50, log=True, show=False, save=file_suffix)
    fix_filename("pca_variance_ratio", file_suffix)
    print("> completed plot of pca components")

    # plot - main pca1/pca2 plots for data
    file_suffix = "mglia-natrev-pca.png"
    file_suffix = "mglia-natrev-pca.svg"
    sc.pl.pca(
        ado,
        components="1,2",
        color=[
            "paper",
            "dataset",
            "organism",
            "mtype",
            "org-mtype",
            "org-mtype-age",
            "ds-seqtype",
        ],
        size=145,
        ncols=1,
        wspace=0.25,
        show=False,
        save=file_suffix,
    )
    fix_filename("pca", file_suffix)
    print("> completed plot of data using pca1 and pca2")

    print("")
    print("done")
    print("")
