from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import json
from tqdm import tqdm

from detectors.car_pose import CarPoseDetector
from opts import opts
import shutil

from tools.evaluation.kitti_utils.eval import kitti_eval
from tools.evaluation.kitti_utils import kitti_common as kitti

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net', 'dec']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.faster=False
    Detector = CarPoseDetector
    detector = Detector(opt)
    print('results_dir',opt.results_dir)
    if os.path.exists(opt.results_dir):
        shutil.rmtree(opt.results_dir,True)
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      if opt.demo[-3:]=='txt':
          with open(opt.demo,'r') as f:
              lines = f.readlines()
          image_names=[os.path.join(opt.data_dir+'/kitti/image/',img.replace('\n','')+'.png') for img in lines]
      else:
        image_names = [opt.demo]
    time_tol = 0
    num = 0
    for _, image_name in enumerate(tqdm(image_names)):
      num+=1
      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
          time_tol=time_tol+ret[stat]
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)

if __name__ == '__main__':
    opt = opts().init()
    checkpoints_path = os.path.join(opt.exp_dir, opt.exp_id)
    AP_path = os.path.join(checkpoints_path, "RTM3D")

    iteration_list = []
    val_mAP = []

    if not os.path.exists(AP_path):
        os.makedirs(AP_path)

    for model_name in os.listdir(checkpoints_path):
        if "pth" not in model_name or "last" in model_name or "best" in model_name:
            continue
        iteration = int(model_name.split(".")[0].split('_')[1])
        iteration_list.append(iteration)
    iteration_list = sorted(iteration_list)
    
    for iteration in iteration_list:
        model_name = "model_{}.pth".format(iteration)
        opt.load_model = os.path.join(checkpoints_path, model_name)
        demo(opt)
        pred_label_path = opt.results_dir + '/data'
        gt_label_path = './kitti_format/data/kitti/training/label_2'
        pred_annos, image_ids = kitti.get_label_annos(pred_label_path, return_ids=True)
        gt_annos = kitti.get_label_annos(gt_label_path, image_ids=image_ids)
        result, ret_dict = kitti_eval(gt_annos, pred_annos, ["Car", "Pedestrian", "Cyclist"])

        if ret_dict is not None:
            mAP_3d_moderate = ret_dict["KITTI/Car_3D_moderate_strict"]
            val_mAP.append(mAP_3d_moderate)
            with open(os.path.join(AP_path, "val_mAP.json"),'w') as file_object:
                json.dump(val_mAP, file_object)
            with open(os.path.join(AP_path, 'epoch_result_{}_{}.txt'.format(iteration, round(mAP_3d_moderate, 2))), "w") as f:
                f.write(result)

