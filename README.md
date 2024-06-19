# AIRFold

**Features:**

- launch all with one `docker-compose up`
- services run in isolated docker container
- submit tasks with RESTful API (FastAPI)
- separated task queues
- concurrence control
- tasks monitor

## Quick Start

## Installation and running your first prediction

You will need a machine running Linux, AlphaFold does not support other
operating systems. Full installation requires up to 3 TB of disk space to keep
genetic databases (SSD storage is recommended) and a modern NVIDIA GPU (GPUs
with more memory can predict larger protein structures).

Please follow these steps:

1.  Install [Docker](https://www.docker.com/).
    *   Install
        [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
        for GPU support.
    *   Setup running
        [Docker as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

1.  Clone this repository and `cd` into it.

    ```bash
    git clone https://github.com/deepmind/alphafold.git
    cd ./alphafold
    ```

1.  Download genetic databases and model parameters:

    *   Install `aria2c`. On most Linux distributions it is available via the
    package manager as the `aria2` package (on Debian-based distributions this
    can be installed by running `sudo apt install aria2`).

    *   Please use the script `scripts/download_all_data.sh` to download
    and set up full databases. This may take substantial time (download size is
    556 GB), so we recommend running this script in the background:

    ```bash
    scripts/download_all_data.sh <DOWNLOAD_DIR> > download.log 2> download_all.log &
    ```

    *   **Note: The download directory `<DOWNLOAD_DIR>` should *not* be a
    subdirectory in the AlphaFold repository directory.** If it is, the Docker
    build will be slow as the large databases will be copied into the docker
    build context.

    *   It is possible to run AlphaFold with reduced databases; please refer to
    the [complete documentation](#genetic-databases).


1.  Check that AlphaFold will be able to use a GPU by running:

    ```bash
    docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
    ```

    The output of this command should show a list of your GPUs. If it doesn't,
    check if you followed all steps correctly when setting up the
    [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
    or take a look at the following
    [NVIDIA Docker issue](https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-801479573).

    If you wish to run AlphaFold using Singularity (a common containerization
    platform on HPC systems) we recommend using some of the third party Singularity
    setups as linked in https://github.com/deepmind/alphafold/issues/10 or
    https://github.com/deepmind/alphafold/issues/24.

1.  Build the Docker image:

    ```bash
    docker build -f docker/Dockerfile -t alphafold .
    ```

    If you encounter the following error:

    ```
    W: GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC
    E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 InRelease' is not signed.
    ```

    use the workaround described in
    https://github.com/deepmind/alphafold/issues/463#issuecomment-1124881779.

1.  Install the `run_docker.py` dependencies. Note: You may optionally wish to
    create a
    [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html)
    to prevent conflicts with your system's Python environment.

    ```bash
    pip3 install -r docker/requirements.txt
    ```

1.  Make sure that the output directory exists (the default is `/tmp/alphafold`)
    and that you have sufficient permissions to write into it.

1.  Run `run_docker.py` pointing to a FASTA file containing the protein
    sequence(s) for which you wish to predict the structure (`--fasta_paths`
    parameter). AlphaFold will search for the available templates before the
    date specified by the `--max_template_date` parameter; this could be used to
    avoid certain templates during modeling. `--data_dir` is the directory with
    downloaded genetic databases and `--output_dir` is the absolute path to the
    output directory.

    ```bash
    python3 docker/run_docker.py \
      --fasta_paths=your_protein.fasta \
      --max_template_date=2022-01-01 \
      --data_dir=$DOWNLOAD_DIR \
      --output_dir=/home/user/absolute_path_to_the_output_dir
    ```

1.  Once the run is over, the output directory shall contain predicted
    structures of the target protein. Please check the documentation below for
    additional options and troubleshooting tips.

### Genetic databases

This step requires `aria2c` to be installed on your machine.

AlphaFold needs multiple genetic (sequence) databases to run:

*   [BFD](https://bfd.mmseqs.com/),
*   [MGnify](https://www.ebi.ac.uk/metagenomics/),
*   [PDB70](http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/),
*   [PDB](https://www.rcsb.org/) (structures in the mmCIF format),
*   [PDB seqres](https://www.rcsb.org/) – only for AlphaFold-Multimer,
*   [UniRef30 (FKA UniClust30)](https://uniclust.mmseqs.com/),
*   [UniProt](https://www.uniprot.org/uniprot/) – only for AlphaFold-Multimer,
*   [UniRef90](https://www.uniprot.org/help/uniref).

**Launch the demo:**

``` sh
docker-compose up
```

**Check the page:**

- FastAPI page: http://127.0.0.1:8081/docs
- tasks monitor page (powered by [flower](https://github.com/mher/flower)): http://127.0.0.1:5555

*Note: please change IP address and ports accordingly, they are specified in `docker-compose.yml`*
