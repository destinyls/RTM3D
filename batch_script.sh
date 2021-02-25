python ./src/main.py --arch res_18 --batch_size 8 --master_batch_size 8 --lr 1.25e-4 --exp_id KM3D_Light_FPN_res18_007
python ./src/KM3D_evaluation.py --arch res_18 --exp_id KM3D_Light_FPN_res18_007

python ./src/main.py --arch res_18 --batch_size 8 --master_batch_size 8 --lr 1.25e-4 --exp_id KM3D_Light_FPN_res18_007
python ./src/KM3D_evaluation.py --arch res_18 --exp_id KM3D_Light_FPN_res18_007
