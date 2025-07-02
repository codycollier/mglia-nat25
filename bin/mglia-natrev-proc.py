#!/usr/bin/env python3
"""The nature review response for Andi's mglia paper

Adhoc code to load, process, and normalize the 18 datasets from 7-8 papers


Expects a custom tree of files similar to this:

(.venv) [mist data]$tree -L 2 {data_root}/
{data_root}/
├── 1_Li_Neuron_2019
│   ├── GSE123021_BulkRNAseq_mouse_P60_regions_processed_data.csv
│   ├── GSE123022_C57BL6J_TREM2KO_369_Microglia_processed_data.csv
│   ├── GSE123022_metadata.csv
│   ├── GSE123022_work
│   ├── GSE123024_Gpnmb_Clec7a_143_Microglia_processed_data.csv
│   ├── GSE123025-groupings.csv
│   ├── GSE123025_Single_myeloid_1922_cells_processed_data.csv
│   └── GSE123025_work
├── 2_Hammond_Immunity_2019
│   ├── GSE121654-groupings.csv
│   └── GSE121654_work
├── 3_Kracht_Science_2020
│   ├── GSE141862-groupings.txt
│   └── GSE141862_work
---snip---


"""

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import plseq


# configure data locations
data_root = "./mglia-natrev/datasets/"
proc_root = "./mglia-natrev/procdata/"


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def validate_start():
    """Check that program is ready to run"""
    if not os.path.exists(data_root):
        print("")
        print(f"Directory not found: {data_root}")
        print("")
        print(
            "Error: Must run this from directory where data is available on custom path"
        )
        print("")
        sys.exit(1)


def add_meta(ado, meta_paper, meta_group):
    """Add pooled obs name and labels to the anndata object

    meta_paper = ("paper", "dataset", "GSE-id", group_count)
    meta_group = ("group_id", "mouse/human", "age", "region", "real/derived", "single cell/nucleus")
    """
    # give the observation a unique name ex: p01-d01-g100
    obs_name = f"{meta_paper[0]}-{meta_paper[1]}-{meta_group[0]}"
    ado.obs_names = [
        obs_name,
    ]
    ado.obs["fullname"] = obs_name

    # paper/dataset level metadata
    ado.obs["paper"] = meta_paper[0]
    ado.obs["dataset"] = meta_paper[1]
    ado.obs["geo-id"] = meta_paper[2]
    ado.obs["gcount"] = meta_paper[3]

    # group level metadata
    ado.obs["group"] = meta_group[0]
    ado.obs["organism"] = meta_group[1]
    ado.obs["age"] = meta_group[2]
    ado.obs["region"] = meta_group[3]
    ado.obs["mtype"] = meta_group[4]
    ado.obs["seq-type"] = meta_group[5]

    # custom viz metadata
    ado.obs["paper-age"] = f"{meta_paper[0]}-{meta_group[2]}"
    ado.obs["ds-age"] = f"{meta_paper[1]}-{meta_group[2]}"
    ado.obs["ds-mtype"] = f"{meta_paper[1]}-{meta_group[4]}"
    ado.obs["ds-seqtype"] = f"{meta_paper[1]}-{meta_group[5]}"
    ado.obs["org-mtype"] = f"{meta_group[1]}-{meta_group[4]}"
    ado.obs["org-mtype-age"] = f"{meta_group[1]}-{meta_group[4]}-{meta_group[2]}"

    return ado


def show(ado):
    """Show highlights of the new ado"""
    name = ado.obs_names[0]
    print("")
    print(f"------------------------- {name} --------------------------------")
    print(ado)
    print(ado.to_df().head())
    print("")


def build_group_map(pdir, hint_file):
    """Read a csv file and build a map of group to cells or files"""
    groups = {}
    h = os.path.join(pdir, hint_file)
    with open(h, "r") as infile:
        for line in infile:
            parts = line.strip().split(",")
            group = parts[0].strip()
            cell = parts[1].strip()
            if group not in groups:
                groups[group] = [cell, ]  # fmt: skip
            else:
                groups[group].append(cell)
    return groups


# -----------------------------------------------------------------------------
# Main processing functions
# -----------------------------------------------------------------------------


def paper1():
    """Paper 1: datasets/1_Li_Neuron_2019/"""
    pdir = f"{data_root}/1_Li_Neuron_2019"
    ados = []

    # GSE123021 - status: ready
    #
    # proc:
    # - load single data
    # - separate and pool by region (4 groups, 3 replicates each)
    #
    # $head GSE123021_BulkRNAseq_mouse_P60_regions_processed_data.csv
    # Gene,CTX_rep1,CTX_rep2,CTX_rep3,CB_rep1,CB_rep2,CB_rep3,HIP_rep1,HIP_rep2,HIP_rep3,STR_rep1,STR_rep2,STR_rep3
    # 0610005C13Rik,15,16,7,3,21,2,0,3,6,0,31,7
    # 0610007C21Rik,3874,2788,1749,2594,2337,1372,2267,2875,2177,1737,906,2386
    # ...
    #
    data = "GSE123021_BulkRNAseq_mouse_P60_regions_processed_data.csv"
    hint = "GSE123021_groups.csv"
    meta_paper = ("p1", "ds1", "GSE123021", 4)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    f = os.path.join(pdir, data)
    ado_raw = ad.read_csv(f).transpose()
    for g, (region, cells) in enumerate(groups.items()):
        # meta_group = (f"{g}_{region}", "mouse", "P60", f"{region}", "real", "bulk")
        meta_group = (f"{g}_{region}", "mouse", "adult", f"{region}", "real", "bulk")

        ado_raw_group = ado_raw[cells, :].copy()
        pool = np.array([ado_raw_group.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw_group.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE123022 - status: ready
    #
    # proc:
    # - load single data, add vars (meta), drop non-wild
    # - separate and pool by "gate" (aka sorting) (2 groups)
    #
    # (.venv) [mist 1_Li_Neuron_2019]$head GSE123022_metadata.csv
    # title,strain,genotype,tissue,age,gate,plate_id,well_id,pooled_library_names,total_counts,detected_genes,...
    # A1.1001200160, C57BL6J, wild, Brain, P7, CD45lowCD11+, 1001200160, A1, N571_Trem2_P7, 165, 44, N, N, N, N
    # A1.1001200162, C57BL6J, wild, Brain, P7, CD45lowCD11+Gpnmb+Clec7a+, 1001200162, A1, N571_Trem2_P7, 345235,...
    #
    # [mist 1_Li_Neuron_2019]$cat GSE123022_metadata.csv | cut -f3,6 -d"," | sort | uniq
    # genotype,gate
    # TREM2_KO,CD45lowCD11+
    # TREM2_KO,CD45lowCD11+Gpnmb+Clec7a+
    # wild,CD45lowCD11+
    # wild,CD45lowCD11+Gpnmb+Clec7a+
    #
    #
    data = "GSE123022_C57BL6J_TREM2KO_369_Microglia_processed_data.csv"
    meta = "GSE123022_metadata.csv"
    meta_paper = ("p1", "ds2", "GSE123022", 2)

    groups = ("CD45lowCD11+", "CD45lowCD11+Gpnmb+Clec7a+")

    # load the data
    f = os.path.join(pdir, data)
    ado_raw = ad.read_csv(f).transpose()

    # load metadata and then select only wild genotype
    fm = os.path.join(pdir, meta)
    ado_raw.obs = pd.read_csv(fm)
    ado_raw = ado_raw[ado_raw.obs["genotype"].isin(("wild",))].copy()

    # generate ado for each group
    for g, group in enumerate(groups):
        # meta_group = (f"{g}", "mouse", "P7", "brain", "real", "single-cell")
        meta_group = (f"{g}", "mouse", "youth", "brain", "real", "single-cell")

        ado_raw_group = ado_raw[ado_raw.obs["gate"].isin((group,))].copy()
        pool = np.array([ado_raw_group.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw_group.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE123024 - status: ready
    #
    # proc:
    # - load single data file and pool
    #
    #
    data = "GSE123024_Gpnmb_Clec7a_143_Microglia_processed_data.csv"
    meta_paper = ("p1", "ds3", "GSE123024", 1)
    # meta_group = ("0", "mouse", "P7", "CB", "real", "single-cell")
    meta_group = ("0", "mouse", "youth", "CB", "real", "single-cell")

    f = os.path.join(pdir, data)
    ado_raw = ad.read_csv(f).transpose()
    pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt:skip

    ado = ad.AnnData(X=pool, var=ado_raw.var)
    ado = add_meta(ado, meta_paper, meta_group)
    ado = plseq.genes.normalize_gene_vars(ado)

    show(ado)
    ados.append(ado)

    # GSE123025 - status: ready
    # (group by uniqe age + tissue combinations - 13)
    #
    # proc:
    # - build {group: cell_id_list} map from GSE123025-groupings.csv
    # - load all from data, for each group build new ado, pool, add metadata
    # - concat all the pooled observations (with meta)
    #
    #
    hint = "GSE123025_groups.csv"  # line: P60-OB, B10.1001200106
    data = "GSE123025_Single_myeloid_1922_cells_processed_data.csv"
    meta_paper = ("p1", "ds4", "GSE123025", 13)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    f = os.path.join(pdir, data)
    ado_all = ad.read_csv(f).transpose()
    for g, (group, cells) in enumerate(groups.items()):
        age, region = group.split("-")
        # meta_group = (f"{g}_{group}", "mouse", f"{age}", f"{region}", "real", "single-cell")
        if age == "P60":
            meta_group = (
                f"{g}_{group}",
                "mouse",
                "adult",
                f"{region}",
                "real",
                "single-cell",
            )
        elif age == "P7":
            meta_group = (
                f"{g}_{group}",
                "mouse",
                "youth",
                f"{region}",
                "real",
                "single-cell",
            )

        ado_raw = ado_all[cells, :].copy()
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper2():
    """Paper 2: 2_Hammond_Immunity_2019"""
    pdir = f"{data_root}/2_Hammond_Immunity_2019"
    ados = []

    # GSE121654 - status: ready
    #
    # proc:
    # - build {group: data_file_list} from groupings file
    # - for each group, load each group sample file to ado, then pool, add metadata
    # - concat all the pooled observations
    # - save etc
    #
    # group, sample_data_file
    # E14, GSM3442006_E14_F_B10.dge.txt
    # P5a, GSM3442014_P5_M_A1.dge.txt
    #
    # data dir: GSE121654_work/GSM3442006_E14_F_B10.dge.txt, ...
    #
    #
    hint = "GSE121654_groups.csv"
    data_dir = "GSE121654_work"
    meta_paper = ("p2", "ds5", "GSE121654", 8)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    for g, (group, data_files) in enumerate(groups.items()):
        # collect the ado for each file in a given group
        ados_group = []
        for d, data in enumerate(data_files):
            f = os.path.join(pdir, data_dir, data)
            ado_raw = ad.read_csv(f, delimiter="\t").transpose()  # tab delimiter

            pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
            ado = ad.AnnData(X=pool, var=ado_raw.var)

            # make obs names unique
            ado.obs_names = [f"pool-group-{g}_file-{d}"]

            ados_group.append(ado)

        # concat and pool the samples within the group
        age = group
        # meta_group = (f"{g}_{group}", "mouse", f"{age}", "brain", "real", "single-cell")
        # E14 Old P100a P100b P30 P4 P5a P5b
        if age == "E14":
            meta_group = (
                f"{g}_{group}",
                "mouse",
                "devel",
                "brain",
                "real",
                "single-cell",
            )
        elif age in ("P4", "P5a", "P5b", "P30"):
            meta_group = (
                f"{g}_{group}",
                "mouse",
                "youth",
                "brain",
                "real",
                "single-cell",
            )
        elif age in ("P100a", "P100b", "Old"):
            meta_group = (
                f"{g}_{group}",
                "mouse",
                "adult",
                "brain",
                "real",
                "single-cell",
            )

        ado_raw = ad.concat(ados_group)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper3():
    """Paper 3: 3_Kracht_Science_2020"""
    pdir = f"{data_root}/3_Kracht_Science_2020"
    ados = []

    # GSE141862 - status: ready
    # (group by gw+d, or maybe just gw)
    #
    # proc:
    # - build {group: data_file_list} from groupings file
    # - for each group, load each group sample file to ado, then pool, add metadata
    #
    # $head GSE141862_groups.txt
    #  2018-05_12+6, GSM4214857_103591-001-000-001_S1_completeCounts.txt, 1-1
    #  2018-05_12+6, GSM4214858_103591-001-000-002_S2_completeCounts.txt, 1-2
    #  2018-05_12+6, GSM4214859_103591-001-000-003_S3_completeCounts.txt, 1-3
    #
    hint = "GSE141862_groups.csv"
    data_dir = "GSE141862_work"
    meta_paper = ("p3", "ds6", "GSE141862", 20)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    for g, (dfid_age, data_files) in enumerate(groups.items()):
        # collect the ado for each file in a given group
        ados_group = []
        for d, data in enumerate(data_files):
            f = os.path.join(pdir, data_dir, data)
            ado_raw = ad.read_csv(f, delimiter="\t").transpose()  # tab delimiter

            pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
            ado = ad.AnnData(X=pool, var=ado_raw.var)

            # make obs names unique
            ado.obs_names = [f"pool-group-{g}_file-{d}"]
            ados_group.append(ado)

        # concat and pool the samples within the group
        group, age = dfid_age.split("_")
        # meta_group = (f"{g}_{group}", "human", f"{age}", "CNS", "real", "single-cell")
        meta_group = (f"{g}_{group}", "human", "devel", "CNS", "real", "single-cell")

        ado_raw = ad.concat(ados_group)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper4():
    """Paper 4: 4_Bian_Nature_2020"""
    pdir = f"{data_root}/4_Bian_Nature_2020"
    ados = []

    # GSE133345 - status: ready
    #
    # proc:
    # - load age (carnegie stages) and file list from file
    # - for each file, load to ado, pool samples
    #
    # $ head GSE133345_groups.csv
    # CS15, TRM/GSM3906036_Quality_controled_UMI_data_of_CS15_embryo_01.txt
    # CS17, TRM/GSM3906037_Quality_controled_UMI_data_of_CS17_embryo_01.txt
    # CS11, TRM/GSM3906038_Quality_controled_UMI_data_of_CS11_embryo_01.txt
    # ...
    #
    hint = "GSE133345_groups.csv"
    data_dir = "GSE133345_work"
    meta_paper = ("p4", "ds7", "GSE133345", 8)

    # make groups
    groups = build_group_map(pdir, hint)

    # special case - one file per "group"
    # Note: In this case there is NOT a combining cs15a and cs15b. They are the
    # only two matching CS, but they are different regions and separating them
    # is easier
    files = {}
    for age, filelist in groups.items():
        files[age] = filelist[0]

    # process each file (load, pool, save)
    for g, (age, filename) in enumerate(files.items()):
        # meta_group = (f"{g}_{age}", "human", f"{age}", "mixed body", "real", "single-cell")
        meta_group = (
            f"{g}_{age}",
            "human",
            "devel",
            "mixed body",
            "real",
            "single-cell",
        )

        f = os.path.join(pdir, data_dir, filename)
        ado_raw = ad.read_csv(f, delimiter=" ").transpose()  # space delimiter
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE137010 - status: ready
    # (GSE137010 --> GSM4065391 & GSM4065392)
    # TODO - "might need to cluster". try just loading and pooling first?
    #
    # proc:
    # - load each 10x data set
    # - pool the observations, add metadata
    #
    # data dir:
    #   GSE137010_work/yolk_sac/GSM4065391_cs11_barcodes.tsv, GSM4065391_cs11_genes.tsv, GSM4065391_cs11_matrix.mtx
    #   GSE137010_work/yolk_sac/GSM4065392_cs15_barcodes.tsv, GSM4065392_cs15_genes.tsv, GSM4065392_cs15_matrix.mtx
    #
    #
    data = (("CS11", "GSM4065391_cs11_"), ("CS15", "GSM4065392_cs15_"))
    data_dir = "GSE137010_work/yolk_sac/"
    meta_paper = ("p4", "ds8", "GSE137010", 2)

    # process each of the two datasets (no pool for now)
    for g, (age, prefix) in enumerate(data):
        # meta_group = (f"{g}_{age}", "human", f"{age}", "yolk-sac", "real", "single-cell")
        meta_group = (f"{g}_{age}", "human", "devel", "yolk-sac", "real", "single-cell")

        path = os.path.join(pdir, data_dir)
        ado = sc.read_10x_mtx(path=path, prefix=prefix)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper5():
    """Paper 5: 5_Galina_Cell_2021"""
    pdir = f"{data_root}/5_Galina_Cell_2021"
    ados = []

    # DUOS-000151 - status: notready
    #
    #
    # proc:
    # - ...
    # - ...
    # - ...
    #
    # data:
    # Fin.MG_Vahid_Final_seurat_2024.rds - DO NOT USE THIS ONE
    # iPSC_H1_imgl_LIGER_seurat_2024.rds - USE THIS ONE
    #
    # metadata:
    # iPSC_H1_Integration_metadata2024.xlsx - (2 iMG replicates, 3 icw (human) replicates)
    #
    # > head iPSC_H1_Integration_metadata2024.csv
    # ,nUMI,nGene,dataset,bc,replicate,line,treatment,line_treatment,line_replicate,clusters
    # AAACGAACACTGGACC-1,14334,3622,iMGLs_rep1,AAACGAACACTGGACC-1,1,H1,Ctrl,H1_Ctrl,H1_1,1
    #
    # from discussion:
    # - use 5 groups
    # - 2 iMG replicates, 3 icw (human) replicates)
    #
    data = "iPSC_H1_imgl_LIGER_seurat_2024.h5ad"
    # meta = "iPSC_H1_Integration_metadata2024.csv"
    meta_paper = ("p5", "ds9", "DUOS-00015 ", 5)

    # load the data
    f = os.path.join(pdir, data)
    ado_all = ad.read_h5ad(f)

    # load metadata
    # note: meta isn't really needed because the group is in the obs name
    # fm = os.path.join(pdir, meta)
    # df_meta =  pd.read_csv(fm)
    # ado_all.obs = ado_all.obs.merge(how="left", right=df_meta, left_index=True, right_index=True)

    # groups (meta column "datasets")
    groups = ("iCW50036", "iCW50118", "iCW70347", "iMGLs_rep1", "iMGLs_rep2")

    # build
    for group in groups:
        meta_group = (f"{group}", "human", "na", "na", "derived", "single-cell")

        ado_raw = ado_all[ado_all.obs_names.str.contains(group)].copy()
        # note: special shape for unknown reason
        pool = np.array([ado_raw.X.sum(axis=0), ])[0]  # fmt:skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper6():
    """Paper 6: 6_Kampmann_NatureNeuroscience_2022"""
    pdir = f"{data_root}/6_Kampmann_NatureNeuroscience_2022"
    ados = []

    # GSE178317 - status: ready
    # (PBS treatments are the control. use title, not mistakes in metadata.)
    #
    # proc:
    # - load list of non-treatment samples from sample-list file
    # - load everything from the data file
    # - drop treatment cells
    # - group and pool the itf (3) vs the brownjohn (3)
    #
    # > head GSE178317-sample-list.txt
    # itf, iTF-Microglia_Day9_PBS_01
    # itf, iTF-Microglia_Day9_PBS_02
    # itf, iTF-Microglia_Day9_PBS_03
    # brj, BrownJohn-Microglia_Day9_PBS_01
    # brj, BrownJohn-Microglia_Day9_PBS_02
    # brj, BrownJohn-Microglia_Day9_PBS_03
    #
    #
    data = "GSE178317_iTF_Microglia_RNAseq_abundance.csv"
    hint = "GSE178317_groups.csv"
    meta_paper = ("p6", "ds10", "GSE178317", 2)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    f = os.path.join(pdir, data)
    ado_raw = ad.read_csv(f).transpose()
    for g, (group, cells) in enumerate(groups.items()):
        # meta_group = (f"{g}_{group}", "human", "day9", "na", "derived", "bulk")
        meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "bulk")

        ado_raw_group = ado_raw[cells, :].copy()
        pool = np.array([ado_raw_group.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw_group.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE85839 - status: ready
    # (raw - bulk - 16 sample files with one sample each)
    #
    # proc:
    # - get list of group and sample files from sample list
    # - for each group, load and pool
    #
    # grouping:
    # - drop the NPC samples
    # - group like this:
    #   - separate: 2 primary fetal microglia samples as reference,
    #      - keep these separate
    #   - pool: the 5 induced microglia samples grown in basal medium (pMGL1-5)
    #   - pool: the 3 induced microglia samples grown in neural conditioned medium (pMGL1-3+NCM)
    #
    hint = "GSE85839_groups.csv"
    data_dir = "GSE85839_work/GSE85839_RAW/"
    meta_paper = ("p6", "ds11", "GSE85839", 4)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    for g, (group, data_files) in enumerate(groups.items()):
        # collect the ado for each file in a given group
        ados_group = []
        for d, data in enumerate(data_files):
            f = os.path.join(pdir, data_dir, data)
            ado_raw = ad.read_csv(f, delimiter="\t").transpose()  # tab delimiter

            pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
            ado = ad.AnnData(X=pool, var=ado_raw.var)

            # make obs names unique
            ado.obs_names = [f"pool-group-{g}_file-{d}"]
            ados_group.append(ado)

        # custom group metadata
        if "primary" in group:
            # meta_group = (f"{g}_{group}", "human", "fetal", "cns", "real", "bulk")
            meta_group = (f"{g}_{group}", "human", "devel", "cns", "real", "bulk")
        else:
            meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "bulk")

        # concat and pool the samples within the group
        ado_raw = ad.concat(ados_group)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
        pool = (pool / pool.sum()) * 1000000  # FPKM to TPM conversion

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE97744 - status: ready
    # (bulk. dropped macrophages. keep the microglia bulk samples.)
    #
    # proc:
    # - load the desired groups and samples from sample-list
    # - load the single data file with observations
    # - for each grouping, select cells and pool
    #
    # grouping
    # - keep the four human samples separate (4)
    # - pool all the iPSC (1)
    #
    #
    hint = "GSE97744_groups.csv"
    data = "GSE97744_expression.log2.CPM.txt"
    meta_paper = ("p6", "ds12", "GSE97744", 5)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    f = os.path.join(pdir, data)
    ado_raw = ad.read_csv(f, delimiter="\t").transpose()
    for g, (group, cells) in enumerate(groups.items()):
        # custom group metadata
        if "ipsc" in group:
            meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "bulk")
        else:
            # meta_group = (f"{g}_{group}", "human", "fetal", "brain", "real", "bulk")
            meta_group = (f"{g}_{group}", "human", "devel", "brain", "real", "bulk")

        ado_raw_group = ado_raw[cells, :].copy()
        pool = np.array([ado_raw_group.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw_group.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE102335 - status: ready
    # (bulk. load only the microglia. skip ipsc and fibroblast.)
    # (using the fpkm. the deseq2norm has odd magnitudes.)
    #
    # proc:
    # - get list of groups and sample files from sample list
    # - for each group, load files, get ado, pool
    #
    # grouping:
    # - drop the fibroblast and ipsc (no derivation yet)
    # - pMG are primary - keep the three separate (3)
    # - oMG - pool by (5 weeks, 7 weeks) (2) (aka 38 days, 52 days)
    #
    hint = "GSE102335_groups.csv"
    data_dir = "GSE102335_work/GSE102335_PROC"
    meta_paper = ("p6", "ds13", "GSE102335", 5)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    for g, (group, data_files) in enumerate(groups.items()):
        # collect the ado for each file in a given group
        ados_group = []
        for d, data in enumerate(data_files):
            f = os.path.join(pdir, data_dir, data)
            ado_raw = ad.read_csv(f, delimiter="\t").transpose()  # tab delimiter

            pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt:skip
            ado = ad.AnnData(X=pool, var=ado_raw.var)

            # make obs names unique
            ado.obs_names = [f"pool-group-{g}_file-{d}"]
            ado.var_names_make_unique()
            ados_group.append(ado)

        # custom group metadata
        if "hmg" in group:
            meta_group = (f"{g}_{group}", "human", "adult", "brain", "real", "bulk")
        else:
            meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "bulk")

        # concat and pool the samples within the group
        ado_raw = ad.concat(ados_group)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
        pool = (pool / pool.sum()) * 1000000  # FPKM to TPM conversion

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE99074 - status: ready
    #
    # proc:
    # - for each file, load single file, create ado, pool, add metadata
    #
    # grouping / processing:
    # - process sample metadata (0, 1 scripts)
    # - use the human samples, not the mouse
    # - keep microglia (drop the wb - whole brain, brain biopsy, etc)
    # - grouping: just pool all 39 together for now (or maybe M vs F for 2)
    # - error with cell id: S12110100 (correct) vs S1210100 (wrong)
    #
    # data = ("GSE99074_HumanMicrogliaBrainVoomNormalization.txt", "GSE99074_MouseBrainVoomNormalization.txt")
    data = "GSE99074_HumanMicrogliaBrainVoomNormalization.txt"
    hint = "GSE99074_groups.csv"
    meta_paper = ("p6", "ds14", "GSE99074", 2)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    f = os.path.join(pdir, data)
    ado_all = ad.read_csv(f, delimiter="\t").transpose()
    for g, (group, cells) in enumerate(groups.items()):
        meta_group = (f"{g}_{group}", "human", "adult", "WB", "real", "bulk")

        ado_raw = ado_all[cells, :].copy()
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE89189 - status: ready
    # (bulk - kept only microglia - group by type/age)
    #
    # proc:
    # - groups: Adult microglia, Fetal microglia, [human] iPS microglia
    # - build {group: data_file_list} from file-list file
    # - for each group, load each group sample file as ado (weird format)
    # - add metadata to the ado
    # - concat the ado list
    # - save etc
    #
    # > GSE89189-file-list.txt
    # GSM2360252_10318X2.FPKM.txt, iPS microglia, iMGL, None, GSM2360252.txt
    # GSM2360274_A2.FPKM.txt, Fetal microglia, Fetal microglia, None, GSM2360274.txt
    # GSM2360277_A15.FPKM.txt, Adult microglia, Adult microglia, None, GSM2360277.txt
    # GSM2445478_n200_r2.FPKM.txt, human iPS microglia, iMGL, None, GSM2445478.txt
    #
    #
    # grouping:
    # - fix samples.  drop "human iPS microglia" which say "treatment: withdrawal"
    # - keep the three groups (adult, fetal, ipsc)
    #
    hint = "GSE89189_groups.csv"
    data_dir = "GSE89189_work/GSE89189_PROC"
    meta_paper = ("p6", "ds15", "GSE89189", 3)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    for g, (group, data_files) in enumerate(groups.items()):
        # collect the ado for each file in a given group
        ados_group = []
        for d, data in enumerate(data_files):
            f = os.path.join(pdir, data_dir, data)
            ado_raw = ad.read_csv(f, delimiter="\t").transpose()  # tab delimiter

            pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt:skip
            ado = ad.AnnData(X=pool, var=ado_raw.var)

            # make obs names unique
            ado.obs_names = [f"pool-group-{g}_file-{d}"]
            ados_group.append(ado)

        # custom group metadata
        if "ips" in group.lower():
            meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "bulk")
        else:
            # groups: devel, adult, iPS
            meta_group = (
                f"{g}_{group}",
                "human",
                f"{group}",
                "unknown",
                "real",
                "bulk",
            )

        # concat and pool the samples within the group
        ado_raw = ad.concat(ados_group)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
        pool = (pool / pool.sum()) * 1000000  # FPKM to TPM conversion

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper8():
    """Paper 8: 8_Guttikonda_NatureNeuroscience_2021"""
    pdir = f"{data_root}/8_Guttikonda_NatureNeuroscience_2021"
    ados = []

    # GSE139549 - status: ready
    # (select microglia cells. minor grouping.)
    #
    # proc:
    # - load the sample,group list from list file
    # - load the single data file
    # - for each group:
    # - select in cells for group / inclusion
    # - pool the cells
    #
    # > GSE139549-sample-list.txt
    # s_Sa2411_macro_2__Sa2411_macro, hPSC-microglia, GSM4143581.txt
    # s_23_human_primary__primary, Primary human microglia, GSM4143582.txt
    #
    # grouping
    # - group: human primary
    # - group: h1-or-h9, then round or macro (2)
    # - group: sa2411, then round or macro (2)
    #
    data = "GSE139549_counts_scaled_DESeq.txt.fix"
    hint = "GSE139549_groups.csv"
    meta_paper = ("p8", "ds17", "GSE139549", 5)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    f = os.path.join(pdir, data)
    ado_raw = ad.read_csv(f, delimiter="\t").transpose()
    for g, (group, cells) in enumerate(groups.items()):
        # custom group metadata
        if "primary" in group:
            # TODO(cmc) - Single cell is incorrect here, should be bulk
            # meta_group = (f"{g}_{group}", "human", "unknown", "brain", "real", "single-cell")
            meta_group = (f"{g}_{group}", "human", "adult", "brain", "real", "bulk")
        else:
            meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "bulk")

        ado_raw_group = ado_raw[cells, :].copy()
        pool = np.array([ado_raw_group.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw_group.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    # GSE139550 - status: ready
    # (four different samples, 2 x day 4, 2 x day 8. just keeping them separate for now.)
    #
    # proc:
    # - load the samples list from list file
    # - for each sample, load, create ado, pool
    #
    # grouping
    # - four samples
    # - group: keep day 4 separate (2)
    # - group: combine the day 8 samples (1)
    #
    hint = "GSE139550_groups.csv"
    data_dir = "GSE139550_work/GSE139550_RAW"
    meta_paper = ("p8", "ds18", "GSE139550", 3)

    # make groups
    groups = build_group_map(pdir, hint)

    # generate ado for each group
    for g, (group, data_files) in enumerate(groups.items()):
        # collect the ado for each file in a given group
        ados_group = []
        for d, data in enumerate(data_files):
            f = os.path.join(pdir, data_dir, data)
            ado_raw = ad.read_csv(f, delimiter=",")  # note: no transpose on this one

            pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip
            ado = ad.AnnData(X=pool, var=ado_raw.var)

            # special - drop the weird cell id column
            ado = ado[:, ~ado.var_names.isin(["cell_id", ]), ]  # fmt:skip

            # make obs names unique
            ado.obs_names = [f"pool-group-{g}_file-{d}"]
            ados_group.append(ado)

        # custom group metadata
        meta_group = (f"{g}_{group}", "human", "na", "na", "derived", "single-cell")

        # concat and pool the samples within the group
        ado_raw = ad.concat(ados_group)
        pool = np.array([ado_raw.X.sum(axis=0), ])  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    return ados


def paper9():
    """Paper 9: Andi's paper, the data from the paper under review"""
    pdir = f"{data_root}/9_paper_data"
    ados = []

    #
    # Andi:
    # hmg is the human microglia from actual human samples
    #
    # >>> ado_hmg
    # AnnData object with n_obs × n_vars = 17342 × 36472
    #   obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'libraryID', 'sampleID',
    #        'subjectID', 'regionID', 'sort', 'age', 'age_group', 'percent.mt',
    #        'RNA_snn_res.0.4', 'seurat_clusters', 'cluster2', 'cluster1'
    #   uns: 'commands', 'version'
    #   obsm: 'X_pca', 'X_umap'
    #   layers: 'RNA_counts', 'RNA_data'
    #   obsp: 'RNA_nn', 'RNA_snn'
    #
    # >>> ado_hmg.obs["sampleID"].unique()
    # ['h2023001wh', 'h2023002wh', 'h2024002wh', 'h2014028cx', 'h2014028lv',
    #  'h2018003lv', 'h2018003cx', 'h2018023cx', 'h2018023lv']
    #
    # >>> ado_hmg.obs["subjectID"].unique()
    # ['h2023001', 'h2023002', 'h2024002', 'h2014028', 'h2018003', 'h2018023']
    #
    # >>> ado_hmg.obs["regionID"].unique()
    # ['wh', 'cx', 'lv']
    #
    # >>> ado_hmg.obs["sort"].unique()
    # ['al', 'ol', 'mg']
    #
    # >>> ado_hmg.obs["age"].unique()
    # ['gw22', 'gw23', 'pw03', 'pw02', 'gw30']
    #
    # >>> ado_hmg.obs["age_group"].unique()
    # ['gw', 'pw']
    #
    #
    # Pooled / grouped by age.
    #
    # meta_paper = ("paper", "dataset", "GSE-id", group_count)
    # meta_group = ("group_id", "mouse/human", "age", "region", "real/derived")
    #
    data_file = "hmg_prenatal.h5ad"
    meta_paper = ("p9", "ds19", "paper", 5)

    filename = os.path.join(pdir, data_file)
    ado_raw = ad.read_h5ad(filename=filename)

    # group by age
    # groups = ['gw22', 'gw23', 'pw03', 'pw02', 'gw30']  # age
    groups = [
        "h2023001",
        "h2023002",
        "h2024002",
        "h2014028",
        "h2018003",
        "h2018023",
    ]  # subjectID
    for g, group in enumerate(groups):
        # meta_group = (f"{g}_{group}", "human", "{group}", "unknown", "real", "single-cell")
        meta_group = (
            f"{g}_{group}",
            "human",
            "devel",
            "unknown",
            "real",
            "single-cell",
        )

        # ado_raw_group = ado_raw[ado_raw.obs["age"].isin((group, ))].copy()
        ado_raw_group = ado_raw[ado_raw.obs["subjectID"].isin((group,))].copy()

        # drop unused dimension
        pool = np.array([ado_raw_group.X.sum(axis=0), ])[0]  # fmt: skip

        ado = ad.AnnData(X=pool, var=ado_raw_group.var)
        ado = add_meta(ado, meta_paper, meta_group)
        ado = plseq.genes.normalize_gene_vars(ado)

        show(ado)
        ados.append(ado)

    #
    # Andi:
    # control mg is the induced microglia generated in our lab
    #
    # >>> ado_img
    # AnnData object with n_obs × n_vars = 912 × 36601
    #     obs: 'orig.ident', 'nCount_RNA', 'nFeature_RNA', 'sample_type', 'genotype', 'age',
    #          'percent.mt', 'RNA_snn_res.0.5', 'seurat_clusters', 'RNA_snn_res.0.2'
    #     uns: 'commands', 'version'
    #     obsm: 'X_pca', 'X_umap'
    #     layers: 'RNA_counts', 'RNA_data', 'RNA_scale.data'
    #     obsp: 'RNA_nn', 'RNA_snn'
    #
    # >>> ado_img.obs["age"].unique()
    # ['06w']
    #
    # >>> ado_img.obs["genotype"].unique()
    # ['WT']
    #
    # >>> ado_img.obs["sample_type"].unique()
    # ['mg']
    #
    #
    # All pooled together. No grouping, as there's nothing to group by.
    #
    #
    data_file = "control-mg-img.h5ad"
    meta_paper = ("p9", "ds20", "paper", 1)
    meta_group = ("0", "human", "na", "na", "derived", "single-cell")

    filename = os.path.join(pdir, data_file)
    ado_raw = ad.read_h5ad(filename=filename)

    # drop unused dimension
    pool = np.array([ado_raw.X.sum(axis=0), ])[0]  # fmt: skip

    ado = ad.AnnData(X=pool, var=ado_raw.var)
    ado = add_meta(ado, meta_paper, meta_group)
    ado = plseq.genes.normalize_gene_vars(ado)

    show(ado)
    ados.append(ado)

    return ados


if __name__ == "__main__":
    validate_start()

    ado_file = os.path.join(proc_root, "mglia-natrev-proc.h5ad")

    # Run all the processing
    ado_all = []
    ado_all.extend(paper1())
    ado_all.extend(paper2())
    ado_all.extend(paper3())
    ado_all.extend(paper4())
    ado_all.extend(paper5())
    ado_all.extend(paper6())
    ado_all.extend(paper8())
    ado_all.extend(paper9())  # the paper under review

    # Concat and save
    ado = ad.concat(ado_all)
    ado.write(ado_file)

    # ...
    print("------------------------------------------------------------------")
    print(":: saved final anndata")
    print("------------------------------------------------------------------")
    print(ado)
    print("")
    print(ado.to_df().head(10))
    print("")
    print("")
    print("done")
