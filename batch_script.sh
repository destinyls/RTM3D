python ./src/main.py --arch res_18 --batch_size 1 --master_batch_size 1 --lr 1.25e-4 --exp_id KM3D_Light_FPN_Weighted_Loss_res18_001
python ./src/KM3D_evaluation.py --arch res_18 --exp_id KM3D_Light_FPN_Weighted_Loss_res18_001

python ./src/main.py --arch res_18 --batch_size 8 --master_batch_size 8 --lr 1.25e-4 --exp_id KM3D_Light_FPN_Weighted_Loss_res18_002
python ./src/KM3D_evaluation.py --arch res_18 --exp_id KM3D_Light_FPN_Weighted_Loss_res18_002

python ./src/main.py --arch res_18 --batch_size 8 --master_batch_size 8 --lr 1.25e-4 --exp_id KM3D_Light_FPN_Weighted_Loss_res18_003
python ./src/KM3D_evaluation.py --arch res_18 --exp_id KM3D_Light_FPN_Weighted_Loss_res18_003
