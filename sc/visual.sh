#python visualization.py --dataset_root data/ --dataset modelnet40 --item=6778 --split train --encoder foldnet --k 16 --shape sphere --model_path=snapshot/Multi_Joint_fixcls_conditional_aggressive_allclass/models/modelnet40_best.pkl --use_conditional --condition_class 5


#python visualization.py --dataset_root data/ --dataset modelnet40 --item=6778 --split train --encoder foldnet --k 16 --shape sphere --model_path=snapshot/Multi_Joint_Multi_Joint_fixcls_conditional_aggressive_allclass_stack/models/modelnet40_250.pkl --use_conditional --condition_class 3 --use_my_reconstruct



#python visualization.py --dataset_root data/ --dataset modelnet40 --item=6778 --split train --encoder foldnet --k 16 --shape sphere --model_path=snapshot/Multi_Joint_knn_myrec3_3_30/models/modelnet40_250.pkl --use_conditional --condition_class 3 --use_my_reconstruct


python visualization.py --dataset_root data/ --dataset modelnet40 --item=6778 --split train --encoder foldnet --k 16 --shape sphere --model_path=snapshot/Multi_Joint_fixcls_conditional_aggressive_allclass_myrec2/models/modelnet40_250.pkl --use_conditional --condition_class 3 --use_my_reconstruct

#python visualization.py --dataset_root data/ --dataset modelnet40 --item=6778 --split train --encoder foldnet --k 16 --shape sphere --model_path=snapshot/Multi_Joint_knn_myrec2_3_30/models/modelnet40_250.pkl --use_conditional --condition_class 3 --use_my_reconstruct
#python visualization.py --dataset_root data/ --dataset modelnet40 --item=6778 --split train --encoder foldnet --k 16 --shape sphere --model_path=snapshot/Reconstruct_sphere_2048_conditional_myrec_stack/models/modelnet40_80.pkl --use_conditional --condition_class 3 --use_my_reconstruct 
