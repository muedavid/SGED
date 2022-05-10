import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from PIL import Image
import sys
import time
from datetime import datetime

import DataProcessing.data_processing as data_processing
import Nets.backbones as backbones
import Nets.features as features
import Nets.losses as losses
import Nets.metrics as metrics
import Nets.tools as tools

DATA = 'SceneNet'
MODEL = 'CASENet_Focal_Loss_Beta'
OUTPUT_TRAIN = 'outputTrain'
OUTPUT_TEST = 'outputTest'
HALF = True

IMG_SIZE_HEIGHT = 1280
IMG_SIZE_WIDTH = 720
NUM_CLASSES = 3
BATCH_SIZE = 3

#MAX_IMG_TRAIN = 1000
#MAX_IMG_TEST = 200
MAX_IMG_TRAIN = 1500
MAX_IMG_TEST = 300

CACHE = True
SEED = None
#SEED = 5

#BACKBONE = "MobileNetV2"
#BACKBONE_OUTPUT = [1,2,3,output] #keep in mind: list start with 0.
#BACKBONE_WEIGHTS = "imagenet"
#ALPHA = 0.35
BACKBONE = "RESNet50"
BACKBONE_OUTPUT = [0,1,2,4]
BACKBONE_WEIGHTS = "imagenet"
ALPHA = 1
FINE_TUNING = True



#EPOCHS = 30
EPOCHS = 10
fine_tune_epochs = 20

#Model Callback
MODEL_SAVE_EPOCH_FREQ = 5
DEL_CHECKPOINTS_OLD = True

# TENSORBOARD:
TENSORBOARD = False
DEL_OLD_TENSORBOARD = True

# SAVE
SAVE = True

# LOSS
WeightedMultiLabelSigmoidEdgeLoss = False
betaUpper = tf.constant(0.995)
betaLower = tf.constant(0.005)
classWeighted = tf.constant(False)

FocalLoss = True
gamma=tf.constant(4)
alpha=tf.constant(1)
weighted_beta=tf.constant(True)
Beta_upper=tf.constant(0.95)
Beta_lower=tf.constant(0.05)
classWeighted=tf.constant(False)


#TESTING
test = False
if test:
    EPOCHS = 10
    MAX_IMG_TRAIN = 100
    MAX_IMG_TEST = 10

#####################################

tf.random.set_seed(SEED)

paths, files = data_processing.path_definitions(HALF, MODEL, DATA, OUTPUT_TRAIN, OUTPUT_TEST)

data_processing.clean_model_directories(paths, DEL_CHECKPOINTS_OLD, DEL_OLD_TENSORBOARD)

train_ds, img_count_train = data_processing.loader(paths,"TRAIN", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TRAIN)
train_ds = data_processing.dataset_processing(train_ds, cache=True, shuffle= True, batch_size= BATCH_SIZE, prefetch= True, img_count = img_count_train)

test_ds, img_count_test = data_processing.loader(paths,"TEST", IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH, HALF, MAX_IMG_TEST)
test_ds = data_processing.dataset_processing(test_ds, cache=True, shuffle= False, batch_size= BATCH_SIZE, prefetch= True, img_count = img_count_test)

#######################################


backbone, layers, output_names = backbones.get_backbone(name=BACKBONE,weights=BACKBONE_WEIGHTS,
                                          height=IMG_SIZE_HEIGHT,width=IMG_SIZE_WIDTH,
                                          alpha=ALPHA, output_layer = BACKBONE_OUTPUT,
                                          trainable=False, FT=FINE_TUNING)

upsample_side_1 = tf.keras.layers.Conv2D(1, kernel_size=1, strides=(1, 1), padding='same')(layers[0])
upsample_side_2 = features.side_feature_casenet(layers[1],channels=1,kernel_size_transpose=4,stride_transpose=2)
upsample_side_3 = features.side_feature_casenet(layers[2],channels=1,kernel_size_transpose=8,stride_transpose=4)
upsample_side_5 = tf.image.resize(layers[3],(int(IMG_SIZE_HEIGHT/16),int(IMG_SIZE_WIDTH/16)))
upsample_side_5 = features.side_feature_casenet(upsample_side_5,channels=NUM_CLASSES,kernel_size_transpose=16,stride_transpose=8,name='side5')

side_outputs = [upsample_side_1,upsample_side_2,upsample_side_3,upsample_side_5]
concat = features.shared_concatenation(side_outputs,NUM_CLASSES)

output = features.fused_classification(concat,NUM_CLASSES,name="output")

model = tf.keras.Model(inputs = backbone.input, outputs = [output,upsample_side_5])

if WeightedMultiLabelSigmoidEdgeLoss:
    loss = lambda y_true, y_pred : losses.weighted_multi_label_sigmoid_loss(y_true, y_pred, beta_lower=Beta_lower, beta_upper=Beta_upper, classWeighted=classWeighted)
elif FocalLoss:
    loss = lambda y_true, y_pred : losses.focal_loss_edges(y_true, y_pred, gamma=gamma, alpha=alpha, weighted_beta=weighted_beta, Beta_lower=Beta_lower, Beta_upper=Beta_upper, classWeighted=classWeighted)
else:
    raise ValueError("either FocalLoss or WeightedMultiLabelSigmoidLoss must be True")

######################################################################

# same Loss function is applied to both outputs, thus I only pass one loss function

# learning rate schedule
# base_learning_rate = 0.00125
base_learning_rate = 0.015
end_learning_rate = 0.001
decay_step = np.ceil(img_count_train / BATCH_SIZE)*EPOCHS
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(base_learning_rate,decay_steps = decay_step,end_learning_rate = end_learning_rate, power = 0.9)

frequency = int(np.ceil(img_count_train / BATCH_SIZE)*MODEL_SAVE_EPOCH_FREQ)

logdir = os.path.join(paths['TBLOGS'], datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = paths["CKPT"]+ "/ckpt-loss={loss:.2f}-epoch={epoch:.2f}",save_weights_only=False,save_best_only=False,monitor="val_loss",verbose=1,save_freq= frequency),
            tf.keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1)]

# compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=loss,
              metrics={'output': [metrics.BinaryAccuracy_Edges(threshold = tf.constant(0),bs = BATCH_SIZE),
                                 metrics.Recall_Edges(),metrics.Precision_Edges()]})

history = model.fit(train_ds, epochs=EPOCHS, validation_data=test_ds, callbacks=callbacks)

##################################################################

plot_losses = ["output_loss"]
plot_metrics = ["output_accuracy_edges", "output_recall_edges", "output_precision_edges"]

path = os.path.join(paths["FIGURES"],"training.svg")

tools.plot_training_results(res=history.history, losses=plot_losses, metrics=plot_metrics, save=SAVE, path=path)

for img, label in test_ds.take(1):
    img, label = img, label

predictions = model.predict(img)

predictions = tools.predict_class_postprocessing(predictions[0], threshold=0.5)

path = os.path.join(paths["FIGURES"], "images")
tools.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=3)

###################################################################

if FINE_TUNING:

    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(backbone.layers))

    # Fine-tune from this layer onwards
    fine_tune_output = output_names[3 - 1]

    backbone.trainable = True

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in backbone.layers:
        layer.trainable = False
        if layer.name == fine_tune_output:
            break

    total_epochs = EPOCHS + fine_tune_epochs

    base_learning_rate = 0.0075
    end_learning_rate = 0.00001
    decay_step = np.floor(img_count_train / BATCH_SIZE) * fine_tune_epochs
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        base_learning_rate, decay_steps=decay_step, end_learning_rate=end_learning_rate, power=0.9)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss=loss,
                  metrics={'output': [metrics.BinaryAccuracy_Edges(threshold=tf.constant(0), bs=BATCH_SIZE),
                                      metrics.Recall_Edges(), metrics.Precision_Edges()]})

    history_fine = model.fit(train_ds, epochs=total_epochs,
                             initial_epoch=history.epoch[-1] + 1, validation_data=train_ds.take(1),
                             callbacks=callbacks)

    plot_losses = ["output_loss"]
    plot_metrics = ["output_accuracy_edges", "output_recall_edges", "output_precision_edges"]

    path = os.path.join(paths["FIGURES"], "fine_tuning_training.svg")

    tools.plot_training_results(res=history.history, res_fine=history_fine.history,
                                losses=plot_losses, metrics=plot_metrics, save=SAVE, path=path, epochs=EPOCHS)

for img, label in test_ds.take(1):
    img, label = img, label

predictions = model.predict(img)
predictions = tools.predict_class_postprocessing(predictions[0], threshold=0.5)

path = os.path.join(paths["FIGURES"], "fine_tuning_images_0,5")
tools.plot_images(images=img, labels=label, predictions=predictions, save=SAVE, path=path, batch_size=3)

####################################################


if SAVE:
    model.save(paths["MODEL"])

    custom_objects = {"BinaryAccuracy_Edges": metrics.BinaryAccuracy_Edges,
                      "Recall_Edges": metrics.Recall_Edges,
                      "Precision_Edges": metrics.Precision_Edges,
                      "lambda": loss}


def predict_real(path, HEIGHT, WIDTH):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, (IMG_SIZE_HEIGHT, IMG_SIZE_WIDTH))
    pred = model.predict(image)

    val_max = tf.reduce_max(pred[0], axis=-1)
    idx_max = tf.argmax(pred[0], axis=-1) + 1

    predictions = tf.where(val_max >= 0, idx_max, 0)

    predictions = tf.expand_dims(predictions, axis=-1)

    return image, predictions

IMG_SIZE_HEIGHT = 1280
IMG_SIZE_WIDTH = 720

imgs = [1,2,3,4,5, 6]
path = '/home/david/test_images/'
Data = {'img':[],'mask':[]}

for i in imgs:
    img, mask = predict_real(path+str(i)+'.png',IMG_SIZE_HEIGHT,IMG_SIZE_WIDTH)
    Data['img'].append(img)
    Data['mask'].append(mask)

#########################################


plt.figure(figsize=(10, 50))
size = len(Data['img'])
for k in range(size):
    plt.subplot(size,2, k*2+1)
    plt.title("Images")
    plt.imshow(tf.keras.preprocessing.image.array_to_img(Data['img'][k][0,:,:,:]))
    plt.subplot(size,2, k*2+2)
    plt.title("Estimation")
    plt.imshow(Data['mask'][k][0,:,:],cmap='gray')
    plt.axis('off')

if SAVE:
    plt.savefig(os.path.join(paths["FIGURES"],'real.svg'), bbox_inches='tight')
plt.show()


