# change /path/to/stored_matches/ to your path
# if you need SIFT results replace superpoint with sift everywhere
# set -nw to your number of cores, setting -nw 1 runs the code without parallelization (for debugging)

# Synth cases A, B, C (seq) for phototourism and ETH3D
for sarg in $( seq 2 2); do
    for x in $( ls /path/to/stored_matches/phototourism); do
        echo $x
        # different cameras
        python eval.py -s $sarg -nw 64 /path/to/stored_matches/phototourism/$x/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/phototourism/$x
        # same cameras
        python eval.py -e -s $sarg -nw 64 /path/to/stored_matches/phototourism/$x/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/phototourism/$x
    done
done

for sarg in $( seq 2 2); do
    for x in $( ls /path/to/stored_matches/ETH3D/multiview_undistorted); do
        echo $x
        python eval.py -s $sarg -nw 64 /path/to/stored_matches/ETH3D/multiview_undistorted/$x/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/ETH3D/multiview_undistorted/$x
        python eval.py -e -s $sarg -nw 64 /path/to/stored_matches/ETH3D/multiview_undistorted/$x/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/ETH3D/multiview_undistorted/$x
    done
done

# rotunda
python eval.py -nw 64 -e /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/rotunda_new
python eval.py -nw 64 /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/rotunda_new

# rotunda graph
python eval.py -nw 64 -g -e /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/rotunda_new
python eval.py -nw 64 -g /path/to/stored_matches/rotunda_new/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/rotunda_new

# cathedral
python eval.py -nw 64 -e /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/cathedral
python eval.py -nw 64 /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/cathedral

# cathedral
python eval.py -nw 64 -g -e /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG_eq /path/to/stored_matches/cathedral
python eval.py -nw 64 -g /path/to/stored_matches/cathedral/pairs-features_superpoint_noresize_2048-LG /path/to/stored_matches/cathedral