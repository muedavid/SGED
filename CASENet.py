#!/usr/bin/env python
# coding: utf-8

# # Remarks
# 
# * Data normalization
#     * Mobilenet expects data from -1 to 1
#         * Normalize Input Data or Include in Model
#         * TFLite Conversion must fit according to decision
#     * Ground Truth Data: for better inspection Data multiplied by 80. Undo the change in the Data Input Pipeline
# * Overview in Tutorials:
#     * tf.function
#     * Repeat addapted Version of using Build in methods for training, ...
#     * Save models using keras
#         * CaseNet first real model: check in implementation of Frey if a Layer needs to be written
#         * other Example: depth seperable dilated convolution,
# * Idea
#     * Loss
#         * Focal Loss: for imbalanced Data
#         * In general Loss: just now weight in each dependent on number of Edge Pixels

# # Libraries

# In[1]:


#!for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import argparse

import DataProcessing.data_processing as data_processing
import Nets.backbones as backbones
import Nets.features as features
import Nets.losses as losses
import Nets.metrics as metrics
import Nets.visualize as visualize
import Nets.tools as tools


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#np.set_printoptions(threshold=sys.maxsize)


# # Parser

# In[2]:


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=False, default=None)
parser.add_argument('--data', type=str, required=False, default=None)

parser.add_argument('--bs', type=int, required=False, default=None)
parser.add_argument('--idx', type=int, required=False, default=None)
parser.add_argument('--epoch', type=int, required=False, default=None)
parser.add_argument('--noise', type=float, required=False, default=None)

parser.add_argument('--train_model', action='store_true', default=False)
parser.add_argument('--cache', action='store_true', default=False)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--sigmoid', action='store_true', default=False)
parser.add_argument('--focal', action='store_true', default=False)

parser.add_argument('--beta_upper', type=float, required=False, default=None)
parser.add_argument('--gamma', type=float, required=False, default=None)
parser.add_argument('--alpha', type=float, required=False, default=None)

file_name = None
try:
    file_name = __file__
except:
    print("Jupyter Notebook")
       
if file_name is None:
    args = parser.parse_args("")
    args.train_model = True
    args.cache = True
    #args.save = True
    args.save = False
    args.sigmoid = False
    args.focal = True
else:    
    args = parser.parse_args()


# # Options

# In[3]:


# Generall Parameters
MODEL= 'CASENET_FOCAL_LOSS_0.7_g2_a2_random' if args.model is None else args.model
DATA= 'SceneNetFloorTiledTextureRandom' if args.data is None else args.data
#DATA = 'RealRed'
TRAIN_DS ='Train'
TEST_DS = 'Test'
HALF = True

# Dataset Loading Parameters
IMG_SIZE_HEIGHT = 1280
IMG_SIZE_WIDTH = 720
NUM_CLASSES = 3
MAX_IMG_TRAIN = 500
MAX_IMG_TEST = 300
SEED = None
BATCH_SIZE = 3 if args.bs is None else args.bs
CACHE = args.cache
NOISE_STD = 0.0 if args.noise is None else args.noise

# Model Parameters
BACKBONE = "RESNet50"
BACKBONE_OUTPUT = [0,1,2,4]
BACKBONE_WEIGHTS = "imagenet"
ALPHA = 1
FINE_TUNING = False
FINE_TUNE_EPOCHS = 10
TRAINABLE_IDX = 0 if args.idx is None else args.idx # (3-1), as indexing starts at 0
EPOCHS = 80 if args.epoch is None else args.epoch
SAVE = args.save
TRAIN_MODEL = args.train_model

#Model Callback
MODEL_SAVE_EPOCH_FREQ = 5
DEL_OLD_CHECKPOINTS = False
TENSORBOARD = False
DEL_OLD_TENSORBOARD = True

# LOSS
weighted_multi_label_sigmoid_edge_loss = args.sigmoid
focal_loss = args.focal

beta_upper = 0.7 if args.beta_upper is None else args.beta_upper
beta_lower = 1.0 - beta_upper
gamma=2.0 if args.gamma is None else args.gamma 
alpha=2.0 if args.alpha is None else args.alpha
class_weighted = True
weighted_beta=True


# All Pixels have been labeled correctly and thus we don't need to account shifted labels 
# and a protection band around the labels for the calculation of the metrics

# In the work of Frey he mentioned that state of the Art ? is 2% of diagonal. 
# He takes 1%, I sugest to take a threshold of 3 Pixels. 
#I don't think that I made more then 3 Pixel mistake in labeling and tracking. Thus this is 0.4%
THRESHOLD_EDGE_WIDTH_REAL = 2

# Data Augmentation:
aug_param = {"contrast_factor": 0.8, "brightness": 0.2, "hue": 0.05, "saturation": 0.8, "gaussian_value": 0.015,
            "value": 0.1, "strength_spot": 0.5, "blur": False, "sigma": 1.0}

#TESTING
test = False
if test:
    EPOCHS = 10
    MAX_IMG_TRAIN = 18
    MAX_IMG_TEST = 3


# # Load Dataset, Preprocess Images and Dataset

# In[5]:


tf.random.set_seed(SEED)

paths, files = data_processing.path_definitions(HALF, MODEL, DATA, TRAIN_DS, TEST_DS, make_dirs=True)

data_processing.clean_model_directories(paths, DEL_OLD_CHECKPOINTS, DEL_OLD_TENSORBOARD)

if TRAIN_MODEL:
    
    rng = tf.random.Generator.from_seed(123, alg='philox')


    train_ds, img_count_train = data_processing.load_dataset(paths,"TRAIN", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, 
                                                             MAX_IMG_TRAIN, noise_std=NOISE_STD)
    train_ds = data_processing.dataset_processing(train_ds, cache=CACHE, shuffle=True, batch_size=BATCH_SIZE, prefetch=True, 
                                                  img_count=img_count_train, augment=True, rng=rng, aug_param=aug_param)

test_ds, img_count_test = data_processing.load_dataset(paths,"TEST", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, 
                                                       MAX_IMG_TEST, noise_std=None)
test_ds = data_processing.dataset_processing(test_ds, cache=CACHE, shuffle=False, batch_size=BATCH_SIZE, prefetch=True, 
                                             img_count=img_count_test)


# In[6]:


DATA_REAL = 'RealRed'
TRAIN_REAL = 'Train'
TEST_REAL = 'Test'
TEST_HARD_REAL = 'Test Hard'
IMG_ONLY_REAL = 'Img Only'
BS_REAL = 8

paths_real, files_real = data_processing.path_definitions(HALF, MODEL, DATA_REAL, TRAIN_REAL, TEST_REAL, TEST_HARD_REAL, IMG_ONLY_REAL, make_dirs=False)

test_real_ds, img_count_test_real = data_processing.load_dataset(paths_real,"TEST", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TEST)
test_real_ds = data_processing.dataset_processing(test_real_ds, cache=False, shuffle=False, batch_size=BS_REAL, prefetch=False, img_count = img_count_test_real)


# In[8]:


#for image,label in train_ds.take(1):
#        image, mask = image,label
#plt.figure(figsize=(10,20))
#plt.imshow(tf.keras.preprocessing.image.array_to_img(image[0, :, :, :]))


#for image,label in train_ds.take(1):
#    sample_image, sample_mask = image,label
#
#visualize.plot_images(images=sample_image, labels=sample_mask, predictions=None, batch_size=3)


# # Model

# In[7]:


if weighted_multi_label_sigmoid_edge_loss:
    loss = lambda y_true, y_pred : losses.weighted_multi_label_sigmoid_loss(y_true,y_pred,beta_lower=beta_lower,beta_upper=beta_upper, class_weighted=class_weighted)
elif focal_loss:
    loss = lambda y_true, y_pred : losses.focal_loss_edges(y_true, y_pred, gamma=gamma, alpha=alpha, weighted_beta=weighted_beta,beta_lower=beta_lower,beta_upper=beta_upper, class_weighted=class_weighted)
else:
    raise ValueError("either FocalLoss or WeightedMultiLabelSigmoidLoss must be True")
    


# In[8]:


if TRAIN_MODEL:
    backbone, output_names = backbones.get_backbone(name=BACKBONE,weights=BACKBONE_WEIGHTS,
                                              height=IMG_SIZE_HEIGHT,width=IMG_SIZE_WIDTH,
                                              alpha=ALPHA, output_layer = BACKBONE_OUTPUT,
                                                            trainable_idx = TRAINABLE_IDX)

    upsample_side_1 = tf.keras.layers.Conv2D(1, kernel_size=1, strides=(1, 1), padding='same')(backbone.output[0])
    upsample_side_2 = features.side_feature_casenet(backbone.output[1],channels=1,kernel_size_transpose=4,stride_transpose=2)
    upsample_side_3 = features.side_feature_casenet(backbone.output[2],channels=1,kernel_size_transpose=8,stride_transpose=4)
    #upsample_side_5 = tf.image.resize(backbone.output[3],(int(IMG_SIZE_HEIGHT/16),int(IMG_SIZE_WIDTH/16)))
    upsample_side_5 = features.side_feature_casenet(backbone.output[3],channels=NUM_CLASSES,kernel_size_transpose=16,stride_transpose=8,name='side5')

    side_outputs = [upsample_side_1,upsample_side_2,upsample_side_3,upsample_side_5]
    concat = features.shared_concatenation(side_outputs,NUM_CLASSES)

    output = features.fused_classification(concat,NUM_CLASSES,name="output")

    model = tf.keras.Model(inputs = backbone.input, outputs = [output,upsample_side_5])

    # tf.keras.utils.plot_model(model,show_shapes = True,to_file = 'h.png')


# # Compile and Train Model

# In[9]:


if TENSORBOARD:
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', '--logdir /home/david/SemesterProject/Models/CASENet/logs')


# In[10]:


if TRAIN_MODEL:
    # learning rate schedule
    base_learning_rate = 0.0005
    end_learning_rate = 0.0001
    decay_step = np.ceil(img_count_train / BATCH_SIZE)*EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(base_learning_rate,decay_steps = decay_step,end_learning_rate = end_learning_rate, power = 0.9)

    frequency = int(np.ceil(img_count_train / BATCH_SIZE)*MODEL_SAVE_EPOCH_FREQ)+1

    logdir = os.path.join(paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = paths["CKPT"]+ "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1:.4f}",
                                                    save_weights_only=False,save_best_only=False,monitor="val_f1",verbose=1,save_freq= 'epoch', period=5),
                tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)]

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics={'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
                                      metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_real_ds, callbacks=callbacks)


# In[16]:


model_ckpt = os.listdir(paths['CKPT'])

f1_max = 0
for ckpt_name in model_ckpt:
    if float(ckpt_name[-4:]) > f1_max:
        f1_max = float(ckpt_name[-4:])
        model_path = paths['CKPT']+"/"+ckpt_name
        
        print(model_path)

custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges,
                  "F1Edges": metrics.F1Edges,
                  "<lambda>":loss}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# # Plot Results

# In[17]:


if TRAIN_MODEL:
    plot_losses = ["loss", "output_loss"]
    plot_metrics = ["output_accuracy_edges", "f1", "recall", "precision"]

    path = os.path.join(paths["FIGURES"],"training.svg")

    visualize.plot_training_results(res=history.history, losses=plot_losses, metrics=plot_metrics, save=SAVE, path=path)


# In[18]:


### Maximum F1 Score:
if not TRAIN_MODEL:
    step_width = 0.05
    threshold_range = [0.05,0.95]
    threshold_array = np.arange(threshold_range[0],threshold_range[1]+step_width,step_width)
    threshold_array = np.array([0.025, 0.1, 0.2,0.3,0.4,0.45,0.5,0.55,0.6,0.7,0.8, 0.9, 0.975])

    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"threshold_metrics_evaluation_test_ds.svg")
    threshold_f1_max = visualize.plot_threshold_metrics_evaluation_class(model=model, 
                                                                         ds=test_ds,
                                                                         num_classes=NUM_CLASSES,
                                                                         threshold_array=threshold_array, 
                                                                         threshold_edge_width=0, save=SAVE, 
                                                                         path=path_metrics_evaluation_plot)


# In[19]:


if not TRAIN_MODEL:
    i = 0
    for img, label in test_ds.take(4):
        img, label = img, label

        threshold = 0.5

        predictions = model.predict(img)
        predictions = tools.predict_class_postprocessing(predictions[0], threshold=threshold)

        path = os.path.join(paths["FIGURES"],"img_test_threshold_{}_{}".format(threshold,i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=3)

        threshold = threshold_f1_max
        path = os.path.join(paths["FIGURES"],"img_test_ods_{}".format(i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=3)

        i += 1


# # Fine Tuning

# In[ ]:


if FINE_TUNING and TRAIN_MODEL:

    # Fine-tune from this layer onwards
    fine_tune_output = output_names[1-1]

    model.trainable = True

    # Freeze all the layers before the `fine_tune_at` layer: 
    for submodel in model.layers:
        if submodel.name == "base_model":
            for layer in submodel.layers:
                layer.trainable = False
                if layer.name == fine_tune_output:
                    break
    
    
    total_epochs =  EPOCHS + FINE_TUNE_EPOCHS

    base_learning_rate = 0.00001
    end_learning_rate =  0.00001
    decay_step = np.floor(img_count_train / BATCH_SIZE)*FINE_TUNE_EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        base_learning_rate,decay_steps = decay_step,end_learning_rate = end_learning_rate, power = 0.9)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics={'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
                                      metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})
    

    history_fine = model.fit(train_ds, epochs=total_epochs, 
                               initial_epoch=history.epoch[-1]+1,validation_data=train_ds.take(1), 
                               callbacks=callbacks)
    
    plot_losses = ["loss", "output_loss"]
    plot_metrics = ["output_accuracy_edges", "f1", "recall", "precision"]
    
    path = os.path.join(paths["FIGURES"],"fine_tuning_training.svg")
    
    visualize.plot_training_results(res=history.history, res_fine = history_fine.history, 
                                losses=plot_losses, metrics=plot_metrics, save=SAVE, path=path, epochs=EPOCHS)
    
    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"threshold_metrics_evaluation__fine_tune_test_ds.svg")
    visualize.plot_threshold_metrics_evaluation(model=model, ds=test_ds, threshold_array=threshold_array, 
                                        threshold_edge_width=0, save=SAVE, path=path_metrics_evaluation_plot, 
                                        accuracy_y_lim_min = 0.9)
        
    for img, label in test_ds.take(1):
        img, label = img, label

    predictions = model.predict(img)    
    predictions = tools.predict_class_postprocessing(predictions[0], threshold = 0.5)

    path = os.path.join(paths["FIGURES"],"fine_tuning_images_0,5")
    visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=3)


# # Evaluate on Test DS of Real Images

# In[ ]:


DATA_REAL = 'RealRed'
TRAIN_REAL = 'Train'
TEST_REAL = 'Test'
TEST_HARD_REAL = 'Test Hard'
IMG_ONLY_REAL = 'Img Only'
BS_REAL = 8

paths_real, files_real = data_processing.path_definitions(HALF, MODEL, DATA_REAL, TRAIN_REAL, TEST_REAL, TEST_HARD_REAL, IMG_ONLY_REAL, make_dirs=False)

test_real_ds, img_count_test_real = data_processing.load_dataset(paths_real,"TEST", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TEST)
test_real_ds = data_processing.dataset_processing(test_real_ds, cache=False, shuffle=False, batch_size=BS_REAL, prefetch=False, img_count = img_count_test_real)


# ## Metrics Evaluation

# In[ ]:


if not TRAIN_MODEL:
    step_width = 0.025
    threshold_range = [0.025, 0.975]
    threshold_array = np.arange(threshold_range[0],threshold_range[1]+step_width,step_width)

    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"threshold_metrics_evaluation_test_real_edge_threshold_{:.1f}.svg".format(0))
    threshold_f1_max = visualize.plot_threshold_metrics_evaluation_class(model=model, ds=test_real_ds, 
                                                                   num_classes = NUM_CLASSES,
                                                                   threshold_array=threshold_array, 
                                                                   threshold_edge_width=0, save=SAVE, 
                                                                   path=path_metrics_evaluation_plot)


# ## Visual Results

# In[ ]:


if not TRAIN_MODEL:
    for img, label in test_real_ds.take(1):
        img, label = img, label


    threshold = 0.5

    predictions = model.predict(img)    
    predictions = tools.predict_class_postprocessing(predictions[0], threshold=threshold)

    path = os.path.join(paths["FIGURES"],"images_test_real_threshold_{:.2f}".format(threshold))
    visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=8)

    threshold = threshold_f1_max

    predictions = model.predict(img)    
    predictions = tools.predict_class_postprocessing(predictions[0], threshold=threshold)

    path = os.path.join(paths["FIGURES"],"images_test_real_threshold_ods")
    visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=8)


# # Save Model

# In[20]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics={'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
                                      metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})

if SAVE:
    model.save(paths["MODEL"])
    
    custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges,
                      "F1Edges": metrics.F1Edges,
                      "<lambda>":loss}
    
    model = tf.keras.models.load_model(paths["MODEL"], custom_objects=custom_objects)


# # Plot Other, Additional Data.

# # Addtional Elements to Consider in other Projects
# 
# * Data augmentation for small datasets

# In[ ]:




