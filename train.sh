#!/bin/bash

#python3 CASENet.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --noise=0.2 --sigmoid --beta_upper=1.0 --train_model
python3 CASENet.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --noise=0.2 --sigmoid --beta_upper=1.0

#python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2' --data="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0 --train_model
python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2' --data="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0

#python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2' --data="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0 --train_model
python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2' --data="SceneNetFloorTiledTextureRandom" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0

#python3 CASENet.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --noise=0.2 --sigmoid --beta_upper=1.0 --train_model
python3 CASENet.py --model='CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' --data="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --noise=0.2 --sigmoid --beta_upper=1.0

#python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2' --data="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0 --train_model
python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.5_g2_a2' --data="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.5 --alpha=2.0 --gamma=2.0

#python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2' --data="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0 --train_model
python3 CASENet.py --model='CASENET_FOCAL_LOSS_0.8_g2_a2' --data="SceneNetFloorTiledTextureIMG" --cache --epoch=40 --save --noise=0.2 --focal --beta_upper=0.8 --alpha=2.0 --gamma=2.0
