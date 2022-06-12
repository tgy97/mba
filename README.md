# MorphNet

## Dataset
Follow the instruction in https://github.com/antao97/UnsupervisedPointCloudReconstruction.

## Usage
* use scripts in `sc/reconstruct` to train a vanilla generator.
    *   `sh sc/reconstruct/2048_sphere_condition.sh`
* use scripts in `sc/joint` to train a backdoor generator (MorphNet).
    *   `sh sc/joint/fix_clsmodel_aggressive_allclass.sh`
* use scripts in `sc/poison` to train and test a backdoor classifier.
    *   `sh sc/poison/base_at.sh` 







