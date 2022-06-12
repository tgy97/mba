##MorphNet

###Dataset
Follow the instruction in https://github.com/antao97/UnsupervisedPointCloudReconstruction or download from https://disk.pku.edu.cn:443/link/B137D787C01107E9AFEA0306BCCD7707

###Usage
* use scripts in `sc/reconstruct` to train a vanilla generator.
    *   `sh sc/reconstruct/2048_sphere_condition.sh`
* use scripts in `sc/joint` to train a backdoor generator (MorphNet).
    *   `sh sc/joint/fix_clsmodel_aggressive_allclass.sh`
* use scripts in `sc/poison` to train and test a backdoor classifier.
    *   `sh sc/poison/base_at.sh` 







