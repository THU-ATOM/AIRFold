#!/usr/bin/env python
# This is the setting file of the program
import os

rootpath = "/home/casp15/code/MSA/DeepMSA2"
databasesrootpath = os.path.join(rootpath,"database")

para_json = dict(
    # main program parameter
    programrootpath=rootpath,
    
    qMSApkg=os.path.join(rootpath, "bin/qMSA"),
    dMSApkg=os.path.join(rootpath, "bin/dMSA"),
    python_DeepPotential=os.path.join(rootpath, "anaconda3/bin/python"),

    # submit job parameter
    run_type='local',  # 'local' or 'sbatch'
    partition='xxx_cpu',
    gpu_partition='xxx_gpu',
    account='xxx',
    mMSAcpu=10,
    qMSAcpu=10,
    dMSAcpu=10,

    # database parameter 
    # If you modified the following databases with different version
    # please go to the alphafold and alphafold_multimer folders in bin folder
    # change the corresponding databases in run_alphafold_*.sh
    dMSAhhblitsdb=os.path.join(databasesrootpath, 'uniclust30_2017_04/uniclust30_2017_04'),
    dMSAjackhmmerdb=os.path.join(databasesrootpath, 'uniref90/uniref90.fasta'),
    dMSAhmmsearchdb=os.path.join(databasesrootpath, 'metaclust/metaclust.fasta'),
    qMSAhhblitsdb=os.path.join(databasesrootpath, 'UniRef30_2022_02/UniRef30_2022_02'),
    qMSAjackhmmerdb=os.path.join(databasesrootpath, 'uniref90/uniref90.fasta'),
    qMSAhhblits3db=os.path.join(databasesrootpath, 'bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'),
    qMSAhmmsearchdb=os.path.join(databasesrootpath, 'mgnify/mgy_clusters.fasta'),
    mMSAJGI=os.path.join(databasesrootpath, 'JGIclust'),

    # alphafold parameter
    alphafold_pkgdir=os.path.join(rootpath, "bin/alphafold"),
    alphafold_libdir=os.path.join(rootpath, "database"),
    alphafold_env=os.path.join(rootpath, "anaconda3/bin"),
    alphafold_seqcut=0.0,
    alphafold_Nmodels=5,
    alphafold_cpu=4,
    alphafold_mem="45GB",

    # alphafold-multimer parameter
    alphafoldm_pkgdir=os.path.join(rootpath, "bin/alphafold_multimer"),
    alphafoldm_MaxPairs=100,
    joint_MSA_filter=True, # True or False
    alphafoldm_Max_MSA_filter=25,
)
