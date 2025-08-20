
### 1. You can download the checkpoint from:
```
Link: https://pan.baidu.com/s/1vG5n-1f8Dur0NujikcxDuQ?pwd=88wa 
Password: 88wa 
```

### 2.Evaluate the StereoAnything model in KITTI12:
```
python tools/eval.py --cfg_file cfgs/nmrf/nmrf_swint_sceneflow_uniform.yaml \\
--eval_data_cfg_file cfgs/nmrf/kitti12_eval_nmrf.yaml \\
--pretrained_model YourPath/StereoAnything-NMRF_swinT.bin```

- `--cfg_file` The path to the config file.
- `--eval_data_cfg_file` The dataset config you want to eval.
- `--pretrained_model` your pre-trained checkpoint
```


You should get the results:
```
'd1_all': tensor(3.4767), 'epe': tensor(0.7544), 'thres_1': tensor(11.4830), 'thres_2': tensor(5.5176), 'thres_3': tensor(3.7987)
```