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
#         * CaseNet first real model: check implementation of Frey if a Layer needs to be written
#         * other Example: depth seperable dilated convolution,

# # Libraries

# In[1]:


#!for a in /sys/bus/pci/devices/*; do echo 0 | sudo tee -a $a/numa_node; done

import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
#import sys
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
parser.add_argument('--model_loaded', type=str, required=False, default=None)
parser.add_argument('--data', type=str, required=False, default=None)
parser.add_argument('--data_base_model', type=str, required=False, default=None)

parser.add_argument('--bs', type=int, required=False, default=None)
parser.add_argument('--idx', type=int, required=False, default=None)
parser.add_argument('--epoch', type=int, required=False, default=None)

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
    args.train_model = False
    args.cache = True
    args.save = True
    args.sigmoid = False
    args.focal = True
else:    
    args = parser.parse_args()


# # Options

# In[3]:


# Generall Parameters
MODEL= 'CASENET_FOCAL_LOSS_0.5_g2_a2_REAL' if args.model is None else args.model
MODEL_LOADED = 'CASENET_FOCAL_LOSS_0.5_g2_a2' if args.model_loaded is None else args.model_loaded
DATA = 'RealRed' if args.data is None else args.data
DATA_BASE_MODEL_LOADED = 'SceneNetFloorTiledTextureIMG' if args.data_base_model is None else args.data_base_model
#MODEL= 'CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' if args.model is None else args.model
#MODEL_LOADED = 'CASENET_MULTI_WEIGHTED_SIGMOID_LOSS_NO_CLIP' if args.model_loaded is None else args.model_loaded
#DATA = 'RealRed' if args.data is None else args.data
#DATA_BASE_MODEL_LOADED = 'RealRed' if args.data_base_model is None else args.data_base_model

TRAIN_DS = 'Train'
TEST_DS = 'Test'
TEST_HARD_DS = 'Test Hard'
TEST_IMG_DS = 'Test IMG'
HALF = True

# Dataset Loading Parameters
IMG_SIZE_HEIGHT = 1280
IMG_SIZE_WIDTH = 720
NUM_CLASSES = 3
MAX_IMG_TRAIN = 100
MAX_IMG_TEST = 25
SEED = None
BATCH_SIZE = 4 if args.bs is None else args.bs
CACHE = args.cache


# Model Parameters
TRAINABLE_IDX = 2 if args.idx is None else args.idx # (3-1), as indexing starts at 0
EPOCHS = 50 if args.epoch is None else args.epoch
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

beta_upper = 0.5 if args.beta_upper is None else args.beta_upper
beta_lower = 1.0 - beta_upper
gamma=2.0 if args.gamma is None else args.gamma 
alpha=2.0 if args.alpha is None else args.alpha
class_weighted = True
weighted_beta=True

THRESHOLD_EDGE_WIDTH_REAL = 2

# Data Augmentation:
aug_param = {"contrast_factor": 0.7, "brightness": 0.05, "hue": 0.02, "saturation": 0.7, "gaussian_value": 0.00,
            "value": 0.05, "strength_spot": 0.3, "blur": True, "sigma": 1.0}

#TESTING
test = False
if test:
    EPOCHS = 10
    MAX_IMG_TRAIN = 100
    MAX_IMG_TEST = 10


# # Load Dataset, Preprocess Images and Dataset

# In[4]:


tf.random.set_seed(SEED)

paths, files = data_processing.path_definitions(HALF, MODEL, DATA, TRAIN_DS, TEST_DS, TEST_HARD_DS, TEST_IMG_DS, 
                                                MODEL_LOADED, DATA_BASE_MODEL_LOADED, make_dirs=True)

data_processing.clean_model_directories(paths, DEL_OLD_CHECKPOINTS, DEL_OLD_TENSORBOARD)

rng = tf.random.Generator.from_seed(123, alg='philox')
train_ds, img_count_train = data_processing.load_dataset(paths,"TRAIN", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TRAIN)
train_ds = data_processing.dataset_processing(train_ds, cache=CACHE, shuffle=False, batch_size=BATCH_SIZE, 
                                              prefetch=True, img_count=img_count_train, augment=True, rng=rng, 
                                              aug_param=aug_param)

#train_ds_2, img_count_train = data_processing.load_dataset(paths,"TRAIN", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TRAIN)
#train_ds_2 = data_processing.dataset_processing(train_ds_2, cache=CACHE, shuffle=True, batch_size=BATCH_SIZE, 
#                                              prefetch=True, img_count=img_count_train, augment=False)


test_ds, img_count_test = data_processing.load_dataset(paths,"TEST", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TEST)
test_ds = data_processing.dataset_processing(test_ds, cache=True, shuffle = False, batch_size=BATCH_SIZE, prefetch=True, img_count=img_count_test, augment=False)

test_hard_ds, img_count_test_hard = data_processing.load_dataset(paths,"TEST_HARD", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TEST)
test_hard_ds = data_processing.dataset_processing(test_hard_ds, cache=False, shuffle=False, batch_size=BATCH_SIZE,
                                                  prefetch=False, img_count=img_count_test_hard, augment=False)

test_img_ds, img_count_test_img = data_processing.load_dataset(paths,"IMG_ONLY", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TEST, has_mask=False)
test_img_ds = data_processing.dataset_processing(test_img_ds, cache=False, shuffle=False, batch_size=6, 
                                                 prefetch=False, img_count=img_count_test_img, augment=False)

if not os.path.exists(os.path.join(paths["FIGURES"],"PRE_TRAINING")):
    os.makedirs(os.path.join(paths["FIGURES"],"PRE_TRAINING"))
    os.makedirs(os.path.join(paths["FIGURES"],"POST_TRAINING"))


# In[5]:


if TRAIN_MODEL:

    for image,label in train_ds.take(1):
        sample_image, sample_mask = image,label

        #visualize.plot_images(images=sample_image, labels=sample_mask, predictions=None, save=False, batch_size=BATCH_SIZE)
        visualize.plot_images(images=sample_image, labels=None, predictions=None, save=False, batch_size=BATCH_SIZE)

    #for image,label in train_ds_2.take(2):
    #    sample_image, sample_mask = image,label
    #
    #    #visualize.plot_images(images=sample_image, labels=sample_mask, predictions=None, save=False, batch_size=BATCH_SIZE)
    #    visualize.plot_images(images=sample_image, labels=None, predictions=None, save=False, batch_size=BATCH_SIZE)


# # Model

# In[6]:


if weighted_multi_label_sigmoid_edge_loss:
    loss = lambda y_true, y_pred : losses.weighted_multi_label_sigmoid_loss(y_true,y_pred,beta_lower=beta_lower,beta_upper=beta_upper, class_weighted=class_weighted)
elif focal_loss:
    loss = lambda y_true, y_pred : losses.focal_loss_edges(y_true, y_pred, gamma=gamma, alpha=alpha, weighted_beta=weighted_beta,beta_lower=beta_lower,beta_upper=beta_upper, class_weighted=class_weighted)
else:
    raise ValueError("either FocalLoss or WeightedMultiLabelSigmoidLoss must be True")
    

custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges,
                  "F1Edges": metrics.F1Edges,
                  "<lambda>":loss}

model = tf.keras.models.load_model(paths["MODEL LOADED"], custom_objects=custom_objects)


# # Metric Numerical Results before Fine Tuning on Real World Data

# ## test

# In[ ]:





# In[7]:


if not TRAIN_MODEL:

    step_width = 0.025
    threshold_range = [0.025,0.975]
    threshold_array = np.arange(threshold_range[0],threshold_range[1]+step_width,step_width)

    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"PRE_TRAINING", "metric_test_threshold{:.1f}.svg".format(0))
    threshold_MF_test = visualize.plot_threshold_metrics_evaluation_class(model=model, 
                                                                          ds=test_ds, 
                                                                          num_classes=NUM_CLASSES, 
                                                                          threshold_array=threshold_array, 
                                                                          threshold_edge_width=0, 
                                                                          save=SAVE, 
                                                                          path=path_metrics_evaluation_plot)

    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"PRE_TRAINING", "metric_test_threshold{:.1f}.svg".format(THRESHOLD_EDGE_WIDTH_REAL))
    visualize.plot_threshold_metrics_evaluation_class(model=model, 
                                                      ds=test_ds,
                                                      num_classes=NUM_CLASSES,
                                                      threshold_array=threshold_array, 
                                                      threshold_edge_width=THRESHOLD_EDGE_WIDTH_REAL, 
                                                      save=SAVE, 
                                                      path=path_metrics_evaluation_plot)


# In[ ]:





# ## test hard

# if not TRAIN_MODEL:
#     step_width = 0.05
#     threshold_range = [0.05,0.95]
#     threshold_array = np.arange(threshold_range[0],threshold_range[1]+step_width,step_width)
# 
#     path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"PRE_TRAINING", "metric_test_hard_threshold{:.1f}.svg".format(0))
#     threshold_MF_test_hard = visualize.plot_threshold_metrics_evaluation_class(model=model, 
#                                                                           ds=test_hard_ds, 
#                                                                           num_classes=NUM_CLASSES, 
#                                                                           threshold_array=threshold_array, 
#                                                                           threshold_edge_width=0, 
#                                                                           save=SAVE, 
#                                                                           path=path_metrics_evaluation_plot)
# 
#     path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"PRE_TRAINING", "metric_test_hard_threshold{:.1f}.svg".format(THRESHOLD_EDGE_WIDTH_REAL))
#     visualize.plot_threshold_metrics_evaluation_class(model=model, 
#                                                       ds=test_ds,
#                                                       num_classes=NUM_CLASSES,
#                                                       threshold_array=threshold_array, 
#                                                       threshold_edge_width=THRESHOLD_EDGE_WIDTH_REAL, 
#                                                       save=SAVE, 
#                                                       path=path_metrics_evaluation_plot)

# # Visual Results before Fine Tuning on Real World Data

# ## test dataset

# In[9]:


if not TRAIN_MODEL:
    i = 1

    for img, label in test_ds.take(3):
        img, label = img, label

        threshold = 0.5

        predictions = model.predict(img)
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"PRE_TRAINING", "images_test_threshold{:.2f}_{}".format(threshold,i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        threshold = threshold_MF_test

        print(threshold_MF_test)
        
        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"PRE_TRAINING", "images_test_ods_{}".format(i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        i += 1


# ## test hard

# In[10]:


if not TRAIN_MODEL:
    i = 1
    for img, label in test_hard_ds.take(2):
        img, label = img, label

        threshold = 0.5

        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"PRE_TRAINING", "images_test_hard_threshold{:.2f}_{}".format(threshold, i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        threshold = threshold_MF_test

        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"PRE_TRAINING", "images_test_hard_ods_{}".format(threshold, i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        i += 1


# ## test img

# In[11]:


if not TRAIN_MODEL:
    for img in test_img_ds.take(1):
        img = img

    predictions = model.predict(img)    
    predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold_MF_test)

    path = os.path.join(paths["FIGURES"],"PRE_TRAINING", "images_test_img")
    try:
        visualize.plot_images(images=img, labels=None, predictions=predictions, save=SAVE, path=path, batch_size=6)
    except:
        visualize.plot_images(images=img, labels=None, predictions=predictions, save=SAVE, path=path, batch_size=img_count_test_img)


# # Compile and Train Model

# In[12]:


if TENSORBOARD:
    get_ipython().run_line_magic('load_ext', 'tensorboard')
    get_ipython().run_line_magic('tensorboard', '--logdir /home/david/SemesterProject/Models/CASENet/logs')


# In[13]:


if TRAIN_MODEL:
    # Fine-tune from this layer onwards
    output_names = ["conv1_relu", "conv2_block3_out", "conv3_block4_out", "conv4_block23_out", "conv5_block3_out"]
    fine_tune_output = output_names[TRAINABLE_IDX]

    model.trainable = True

    # Freeze all the layers before the `fine_tune_at` layer: 
    for submodel in model.layers:
        if submodel.name == "base_model":
            for layer in submodel.layers[0:100]:
                layer.trainable = False
                if layer.name == fine_tune_output:
                    break


# In[14]:


if TRAIN_MODEL:         

    # learning rate schedule
    base_learning_rate = 0.0005
    end_learning_rate = 0.00005
    decay_step = np.ceil(img_count_train / BATCH_SIZE)*EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(base_learning_rate,decay_steps = decay_step,end_learning_rate = end_learning_rate, power = 0.9)

    frequency = int(np.ceil(img_count_train / BATCH_SIZE)*MODEL_SAVE_EPOCH_FREQ)+1

    logdir = os.path.join(paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = paths["CKPT"]+ "/ckpt-loss={val_loss:.2f}-epoch={epoch:.2f}-f1={val_f1:.4f}",
                                                    save_weights_only=False,save_best_only=False,monitor="val_f1",verbose=1,save_freq= 'epoch', period=10),
                tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)]

    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss={'output':loss},
                  metrics={'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
                           metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})

    evaluate_test = model.evaluate(test_ds, return_dict = True, batch_size = 3)
    evaluate_train = model.evaluate(train_ds, return_dict = True, batch_size = 3)

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=callbacks, verbose = 2)

    dict_keys = ['output_loss', 'output_accuracy_edges', 'f1', 'precision', 'recall']
    for key in dict_keys:
        history.history[key] = [evaluate_train[key]] + history.history[key]
        history.history['val_'+key] = [evaluate_test[key]] + history.history['val_'+key]


# In[15]:


model_ckpt = os.listdir(paths['CKPT'])

f1_max = 0
for ckpt_name in model_ckpt:
    if float(ckpt_name[-4:]) > f1_max:
        f1_max = float(ckpt_name[-4:])
        model_path = paths['CKPT']+"/"+ckpt_name

custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges,
                  "F1Edges": metrics.F1Edges,
                  "<lambda>":loss}

model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# # training evolution

# In[16]:


if TRAIN_MODEL:
    plot_losses = ["loss", "output_loss"]
    plot_metrics = ["output_accuracy_edges", "f1", "recall", "precision"]

    path = os.path.join(paths["FIGURES"],"training.svg")

    visualize.plot_training_results(res=history.history, losses=plot_losses, metrics=plot_metrics, save=SAVE, path=path)


    for img, label in train_ds.take(1):
        img, label = img, label

    predictions = model.predict(img)    
    predictions = tools.predict_class_postprocessing(predictions[0], threshold = 0.5)

    visualize.plot_images(images=img, labels=label, predictions=predictions, save=False, path=path, batch_size=BATCH_SIZE)


# # Metric Numerical Results After Fine Tuning on Real World Data

# ## test

# In[17]:


if not TRAIN_MODEL:
    step_width = 0.05
    threshold_range = [0.05,0.95]
    threshold_array = np.arange(threshold_range[0],threshold_range[1]+step_width,step_width)

    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"POST_TRAINING", "metric_test_threshold{:.1f}.svg".format(0))
    threshold_MF_test = visualize.plot_threshold_metrics_evaluation_class(model=model, 
                                                                          ds=test_ds,
                                                                          num_classes = NUM_CLASSES,
                                                                          threshold_array=threshold_array,
                                                                          threshold_edge_width=0, 
                                                                          save=SAVE, 
                                                                          path=path_metrics_evaluation_plot)

    path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"POST_TRAINING", "metric_test_threshold{:.1f}.svg".format(THRESHOLD_EDGE_WIDTH_REAL))
    visualize.plot_threshold_metrics_evaluation_class(model=model, 
                                                ds=test_ds,
                                                num_classes = NUM_CLASSES,
                                                threshold_array=threshold_array, 
                                                threshold_edge_width=THRESHOLD_EDGE_WIDTH_REAL, 
                                                save=SAVE, 
                                                path=path_metrics_evaluation_plot)


# ## test hard

# if not TRAIN_MODEL:
#     step_width = 0.05
#     threshold_range = [0.05,0.95]
#     threshold_array = np.arange(threshold_range[0],threshold_range[1]+step_width,step_width)
# 
#     path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"POST_TRAINING", "metric_test_hard_threshold{:.1f}.svg".format(0))
#     threshold_MF_test_hard = visualize.plot_threshold_metrics_evaluation_class(model=model, 
#                                                                          ds=test_hard_ds,
#                                                                          num_classes = NUM_CLASSES,
#                                                                          threshold_array=threshold_array, 
#                                                                          threshold_edge_width=0, 
#                                                                          save=SAVE, 
#                                                                          path=path_metrics_evaluation_plot)
# 
#     path_metrics_evaluation_plot = os.path.join(paths["FIGURES"],"POST_TRAINING", "metric_test_hard_threshold{:.1f}.svg".format(THRESHOLD_EDGE_WIDTH_REAL))
#     visualize.plot_threshold_metrics_evaluation_class(model=model, 
#                                                 num_classes = NUM_CLASSES
#                                                 ds=test_hard_ds, 
#                                                 threshold_array=threshold_array, 
#                                                 threshold_edge_width=THRESHOLD_EDGE_WIDTH_REAL, 
#                                                 save=SAVE, 
#                                                 path=path_metrics_evaluation_plot)

# # Visual Results After Fine Tuning on Real World Data

# ## test

# In[19]:


if not TRAIN_MODEL:
    i = 1
    for img, label in test_ds.take(3):
        img, label = img, label

        threshold = 0.5
        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"POST_TRAINING", "images_test_threshold_{:.2f}_{}".format(threshold, i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        threshold = threshold_MF_test
        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"POST_TRAINING", "images_test_threshold_ods_{}".format(i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        i += 1


# ## test hard

# In[20]:


if not TRAIN_MODEL:
    i = 1
    for img, label in test_hard_ds.take(2):
        img, label = img, label

        threshold = 0.5

        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"POST_TRAINING", "images_test_hard_threshold{:.2f}_{}".format(threshold,i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        threshold = threshold_MF_test

        predictions = model.predict(img)    
        predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

        path = os.path.join(paths["FIGURES"],"POST_TRAINING", "images_test_hard_ods_{}".format(i))
        visualize.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=BATCH_SIZE)

        i += 1


# ## test img

# In[21]:


if not TRAIN_MODEL:
    for img in test_img_ds.take(1):
        img = img

    predictions = model.predict(img)
    threshold = threshold_MF_test
    predictions = tools.predict_class_postprocessing(predictions[0], threshold = threshold)

    path = os.path.join(paths["FIGURES"],"POST_TRAINING", "images_test_img")

    try:
        visualize.plot_images(images=img, labels=None, predictions=predictions, save=SAVE, path=path, batch_size=6)
    except:
        visualize.plot_images(images=img, labels=None, predictions=predictions, save=SAVE, path=path, batch_size=img_count_test_img)


# # Save Model

# In[ ]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics={'output': [metrics.BinaryAccuracyEdges(threshold_prediction=0),
                                      metrics.F1Edges(threshold_prediction=0, threshold_edge_width=0)]})

if SAVE and TRAIN_MODEL:
    model.save(paths["MODEL"])
    
    custom_objects = {"BinaryAccuracyEdges": metrics.BinaryAccuracyEdges,
                      "F1Edges": metrics.F1Edges,
                      "<lambda>":loss}
    
    model = tf.keras.models.load_model(paths["MODEL"], custom_objects=custom_objects)


# # Addtional Elements to Consider in other Projects
# 
# * Data augmentation for small datasets

# In[ ]:





# In[ ]:




