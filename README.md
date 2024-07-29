# Evaluation of radial distortion solvers

This repo contains code for paper "Are Minimal Radial Distortion Solvers Necessary for Relative Pose Estimation?" (arxiv: TBA, doi: TBA)

## Installation

Create an environment with pytorch and packaged from `requirements.txt`.

Install [PoseLib fork with robust radial distortion estimators](https://github.com/kocurvik/PoseLib/tree/rd) into the environment:
```shell
git clone https://github.com/kocurvik/PoseLib
git cd PoseLib
git checkout rd
pip install .
```

Before running the python scripts make sure that the repo is in your python path (e.g. `export PYTHONPATH=/path/to/repo/solvers_rd`)

## Running experiments

### Datasets
We use four datasets:
* Rotunda - download TBA
* Cathedral - download TBA
* Phototourism - download from the [IMC2020 challenge website](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)
* ETH3D - download the multiview undistorted train data from the [dataset website](https://www.eth3d.net/datasets#high-res-multi-view-training-data).

### Extracting matches

You can download the files with the [matches](http://cogsci.dai.fmph.uniba.sk/~kocur/rd_all_matches.tar.gz).

If you want to extract them yourself you can use `prepare_im.py` and `prepare_bundler.py`.

To prepare matches for all datasets you can run:
```shell
# generates all matches
python prepare_bundler.py -f superpoint /path/to/stored_matches/rotunda_new /path/to/dataset/rotunda_new
# generates only matches with the same camera in both views
python prepare_bundler.py -f superpoint -e /path/to/stored_matches/rotunda_new /path/to/dataset/rotunda_new

python prepare_bundler.py -n 10000 -f superpoint /path/to/stored_matches/cathedral /path/to/dataset/cathedral
python prepare_bundler.py -n 10000 -f superpoint -e /path/to/stored_matches/cathedral /path/to/dataset/cathedral

python prepare_im.py -f superpoint /path/to/stored_matches/ETH3D/multiview_undistorted /path/to/dataset/ETH3D/multiview_undistorted
python prepare_im.py -f superpoint -n 5000 /path/to/stored_matches/phototourism /path/to/dataset/phototourism/
```

You can change the features to SIFT by setting `-f sift`. You can choose any `/path/to/stored_matches/` it can also be the same as the original dataset directory.

### Running experiments

To run the experiments you should modify `experiments.sh` as described in the script comments and run it.

### Tables and figures

Tables and figures are generated using `utils/tables.py`, `utils/tables_real.py` and `utils/vis.py`.

## Citation
TBA