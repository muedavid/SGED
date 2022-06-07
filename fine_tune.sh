#!/bin/bash

#python3 REAL.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.5_g2_a2' --data_base_model="SceneNetFloorTiledTextureIMG" --cache --epoch=60 --save --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0 --train_model
python3 REAL.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.5_g2_a2' --data_base_model="SceneNetFloorTiledTextureIMG" --cache --epoch=60 --save --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0

#python3 REAL.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.8_g2_a2' --data_base_model="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0 --train_model
python3 REAL.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.8_g2_a2' --data_base_model="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0
#
#python3 REAL.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP_REAL' --model_loaded='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data_base_model="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --sigmoid --beta_upper=1.0 --train_model
python3 REAL.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP_REAL' --model_loaded='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data_base_model="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --sigmoid --beta_upper=1.0
#
#python3 REAL.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.5_g2_a2' --data_base_model="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0 --train_model
python3 REAL.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.5_g2_a2' --data_base_model="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0
#
#python3 REAL.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.8_g2_a2' --data_base_model="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0 --train_model
python3 REAL.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.8_g2_a2' --data_base_model="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0
#
#python3 REAL.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP_REAL' --model_loaded='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data_base_model="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --sigmoid --beta_upper=1.0 --train_model
python3 REAL.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP_REAL' --model_loaded='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data_base_model="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --sigmoid --beta_upper=1.0
#
#python3 REAL.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP_REAL' --model_loaded='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data_base_model="SceneNetFloorRandomTextureRandom" --cache --epoch=80 --save --sigmoid --beta_upper=1.0 --train_model
python3 REAL.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP_REAL' --model_loaded='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data_base_model="SceneNetFloorRandomTextureRandom" --cache --epoch=80 --save --sigmoid --beta_upper=1.0
#
#python3 REAL.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.5_g2_a2' --data_base_model="SceneNetFloorRandomTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0 --train_model
python3 REAL.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.5_g2_a2' --data_base_model="SceneNetFloorRandomTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0
#
#python3 REAL.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.8_g2_a2' --data_base_model="SceneNetFloorRandomTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0 --train_model
python3 REAL.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2_REAL' --model_loaded='CASENET_FOCAL_LOSS_0.8_g2_a2' --data_base_model="SceneNetFloorRandomTextureRandom" --cache --epoch=40 --save --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0
