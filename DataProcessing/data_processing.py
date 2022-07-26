import tensorflow as tf
import tensorflow_addons as tfa
import os
import os.path as osp
from glob import glob
import shutil
import numpy as np
import DataProcessing.spot_light as spot_light


def path_definitions(half, model, data, train, test, test_hard=None, img_only=None, model_loaded=None,
                     data_model_loaded=None, make_dirs=False):
    base_path_model = '/home/david/SemesterProject/Models'
    base_path_data = '/home/david/SemesterProject/Datasets'

    if data_model_loaded is None and model_loaded is not None:
        raise ValueError("Define the Dataset used to train the loaded model")

    paths = {
        'MODEL': osp.join(base_path_model, data, model) if data_model_loaded is None
        else osp.join(base_path_model, data_model_loaded, model),
        'CKPT': osp.join(base_path_model, data, model, 'CKPT') if data_model_loaded is None
        else osp.join(base_path_model, data_model_loaded, model, 'CKPT'),
        'TBLOGS': osp.join(base_path_model, data, model, 'logs') if data_model_loaded is None
        else osp.join(base_path_model, data_model_loaded, model, 'logs'),
        'TFLITE': osp.join(base_path_model, data, model, 'TFLITE') if data_model_loaded is None
        else osp.join(base_path_model, data_model_loaded, model, 'TFLITE'),
        'FIGURES': osp.join(base_path_model, data, model, 'FIGURES') if data_model_loaded is None
        else osp.join(base_path_model, data_model_loaded, model, 'FIGURES'),
        'MODEL LOADED': osp.join(base_path_model, data_model_loaded, model_loaded) if type(
            model_loaded) == str else None,
        'DATA': {'TRAIN': osp.join(base_path_data, data, train, 'half' * half + (1 - half) * 'full'),
                 'TEST': osp.join(base_path_data, data, test, 'half' * half + (1 - half) * 'full'),
                 'TEST_HARD': osp.join(base_path_data, data, test_hard,
                                       'half' * half + (1 - half) * 'full') if type(test_hard) == str else None,
                 'IMG_ONLY': osp.join(base_path_data, data, img_only,
                                      'half' * half + (1 - half) * 'full') if type(img_only) == str else None},
        'IMAGE': {'TRAIN': osp.join(base_path_data, data, train, 'half' * half + (1 - half) * 'full', 'images'),
                  'TEST': osp.join(base_path_data, data, test, 'half' * half + (1 - half) * 'full', 'images'),
                  'TEST_HARD': osp.join(base_path_data, data, test_hard, 'half' * half + (1 - half) * 'full',
                                        'images') if type(test_hard) == str else None,
                  'IMG_ONLY': osp.join(base_path_data, data, img_only, 'half' * half + (1 - half) * 'full',
                                       'images') if type(img_only) == str else None},
        'CLASS_ANN': {
            'TRAIN': osp.join(base_path_data, data, train, 'half' * half + (1 - half) * 'full', 'class_annotation'),
            'TEST': osp.join(base_path_data, data, test, 'half' * half + (1 - half) * 'full', 'class_annotation'),
            'TEST_HARD': osp.join(base_path_data, data, test_hard, 'half' * half + (1 - half) * 'full',
                                  'class_annotation') if type(test_hard) == str else None,
            'IMG_ONLY': osp.join(base_path_data, data, img_only, 'half' * half + (1 - half) * 'full',
                                 'class_annotation') if type(img_only) == str else None},
        'INST_ANN': {
            'TRAIN': osp.join(base_path_data, data, train, 'half' * half + (1 - half) * 'full', 'instance_annotation'),
            'TEST': osp.join(base_path_data, data, test, 'half' * half + (1 - half) * 'full', 'instance_annotation'),
            'TEST_HARD': osp.join(base_path_data, data, test_hard, 'half' * half + (1 - half) * 'full',
                                  'instance_annotation') if type(test_hard) == str else None,
            'IMG_ONLY': osp.join(base_path_data, data, img_only, 'half' * half + (1 - half) * 'full',
                                 'instance_annotation') if type(img_only) == str else None},
        'COCO': {'TRAINING': osp.join(base_path_data, data, train, 'half' * half + (1 - half) * 'full', 'coco_data'),
                 'TEST': osp.join(base_path_data, data, test, 'half' * half + (1 - half) * 'full', 'coco_data'),
                 'TEST_HARD': osp.join(base_path_data, data, test_hard, 'half' * half + (1 - half) * 'full',
                                       'coco_data') if type(test_hard) == str else None,
                 'IMG_ONLY': osp.join(base_path_data, data, img_only, 'half' * half + (1 - half) * 'full',
                                      'coco_data') if type(img_only) == str else None},
    }

    files = {
        'OUTPUT_TFLITE_MODEL': osp.join(paths['TFLITE'], model + '.tflite'),
        'OUTPUT_TFLITE_MODEL_METADATA': osp.join(paths['TFLITE'], model + '_metadata.tflite'),
        'OUTPUT_TFLITE_LABEL_MAP': osp.join(paths['TFLITE'], model + '_tflite_label_map.txt'),
    }

    if make_dirs:
        for path in paths.keys():
            if path in ['MODEL', 'CKPT', 'TFLITE', 'TBLOGS', 'FIGURES']:
                if not osp.exists(paths[path]):
                    print(path)
                    os.makedirs(paths[path])

    return paths, files


def parse_image(img_path, has_mask, single_class):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)

    if has_mask:
        mask_path = tf.strings.regex_replace(img_path, "images", "class_annotation")
        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)

        if single_class:
            mask = tf.where(mask >= tf.constant(80, tf.uint8), tf.constant(1, tf.uint8), mask)
        else:
            # For Ground Truth Check, the images are save with values * 80
            mask = tf.where(mask == tf.constant(80, tf.uint8), tf.constant(1, tf.uint8), mask)
            mask = tf.where(mask == tf.constant(160, tf.uint8), tf.constant(2, tf.uint8), mask)
            mask = tf.where(mask == tf.constant(240, tf.uint8), tf.constant(3, tf.uint8), mask)

        return {'image': image, 'mask': mask}
    else:
        return {'image': image}


def resize_label_map(label, current_shape_label, new_shape_label, num_classes):
    # label 3D
    print(label.shape)
    label = tf.cast(label, tf.int32)
    label = tf.expand_dims(label, axis=0)
    class_range = tf.range(1, num_classes + 1)
    class_range_reshape = tf.reshape(class_range, [1, 1, 1, num_classes])
    label_re = tf.cast(class_range_reshape == label, dtype=tf.int32)
    pad = tf.constant([[0, 0], [0, 0], [0, 0], [1, 0]])
    label_re = tf.pad(label_re, pad, "CONSTANT")

    edge_width_height = int(current_shape_label[0] / new_shape_label[0])
    edge_width_width = int(current_shape_label[1] / new_shape_label[1])
    kernel = tf.ones([edge_width_height, edge_width_width, num_classes + 1, 1], tf.float32)
    label_re = tf.cast(label_re, tf.float32)
    label_re = tf.nn.depthwise_conv2d(label_re, kernel, strides=[1, 1, 1, 1], padding="SAME")
    label_re = tf.cast(tf.clip_by_value(label_re, 0, 1), tf.int32)

    label_re = tf.image.resize(label_re, new_shape_label, method='nearest', antialias=True)
    label_re = tf.math.argmax(label_re, axis=-1, output_type=tf.int32)
    label = tf.expand_dims(label_re, axis=-1)
    label = tf.squeeze(label, axis=0)
    return label


def preprocess(datapoint, current_shape, new_shape, num_classes, half, has_mask):
    # Preprocessing Layer already added into model Pipeline: model expects input of 0-255, 3 channels
    # datapoint['image'] = tf.cast(datapoint['image'], tf.float32) / 127.5 - 1
    # datapoint['image'] = tf.cast(datapoint['image'], tf.float32)
    datapoint['image'] = tf.image.resize(datapoint['image'], new_shape, method='nearest')
    datapoint['image'] = tf.cast(datapoint["image"], tf.uint8)

    if has_mask:
        if half:
            datapoint['mask'] = resize_label_map(datapoint['mask'], (int(current_shape[0]/2), int(current_shape[1]/2)), (int(new_shape[0]/2), int(new_shape[1]/2)), num_classes)
        else:
            datapoint['mask'] = resize_label_map(datapoint['mask'], current_shape, (int(new_shape[0] / 2), int(new_shape[1] / 2)), num_classes)
        # datapoint['mask'] = tf.cast(datapoint['mask'], tf.float32)

        # datapoint['mask'] = tf.image.resize(datapoint['mask'], mask_size, method='nearest')
        datapoint['mask'] = tf.cast(datapoint['mask'], tf.uint8)

        return {'image': datapoint['image'], 'mask': datapoint['mask']}
    else:
        return {'image': datapoint['image']}


def add_noise(datapoint, noise_std, has_mask):
    # not sure if necessary to convert from float again back to uint
    datapoint['image'] = tf.cast(datapoint['image'], tf.float32)
    datapoint['image'] = tf.keras.layers.GaussianNoise(noise_std)(datapoint['image'], training=True)
    datapoint["image"] = tf.clip_by_value(datapoint["image"], 0, 255.0)
    datapoint['image'] = tf.cast(datapoint['image'], tf.uint8)
    if has_mask:
        return {'image': datapoint['image'], 'mask': datapoint['mask']}
    else:
        return {'image': datapoint['image']}


def split_dataset_dictionary(datapoint):
    if len(datapoint.keys()) == 2:
        return datapoint['image'], datapoint['mask']
    else:
        return datapoint['image']


def load_dataset(paths, dataset_name, current_shape, new_shape, num_classes, half, max_img, has_mask=True, noise_std=None, single_class=None):
    files = sorted(glob(osp.join(paths["IMAGE"][dataset_name], "*.png")))
    files_filtered = [x for x in files if int(x[-8:-4]) <= max_img - 1]
    dataset = tf.data.Dataset.from_tensor_slices(files_filtered)
    dataset = dataset.map(lambda x: parse_image(x, has_mask, single_class), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(lambda x: preprocess(x, current_shape, new_shape, num_classes, half, has_mask),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if noise_std:
        dataset = dataset.map(lambda x: add_noise(x, noise_std, has_mask),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    image_count = len(files_filtered)

    print("The {mode} Dataset contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=dataset_name))

    return dataset, image_count


def value_augmentation_spot_light(shape, value, strength_spot):

    shape = shape.numpy()

    uniform_value_diff = np.random.uniform(-value, value)
    mask = np.zeros(shape)
    mask[:, :, 2] = uniform_value_diff
    strength = np.random.uniform(0.0, strength_spot)
    mask_raw = spot_light.generate_spot_light_mask(mask_size=(shape[1], shape[0]))
    mask[:, :, 2] = mask[:, :, 2] + strength * mask_raw
    mask[:, :, 1] = -strength * mask_raw
    return mask


def augment_mapping(datapoint, rng, aug_param):

    if aug_param["blur"]:
        sigma = np.random.uniform(0, aug_param["sigma"])
        datapoint["image"] = tfa.image.gaussian_filter2d(datapoint["image"], (5, 5), sigma)


    seed = rng.make_seeds(2)[0]
    datapoint["image"] = tf.image.stateless_random_contrast(datapoint["image"], aug_param["contrast_factor"],
                                                            1 / aug_param["contrast_factor"], seed)
    seed = rng.make_seeds(2)[0]
    datapoint["image"] = tf.image.stateless_random_brightness(datapoint["image"], aug_param["brightness"], seed)

    seed = rng.make_seeds(2)[0]
    datapoint["image"] = tf.image.stateless_random_hue(datapoint["image"], aug_param["hue"], seed)
    seed = rng.make_seeds(2)[0]
    datapoint["image"] = tf.image.stateless_random_saturation(datapoint["image"], aug_param["saturation"],
                                                              1 / aug_param["saturation"], seed)
    seed = rng.make_seeds(2)[0]

    # convert to HSV
    datapoint["image"] = tf.image.convert_image_dtype(datapoint["image"], tf.float32)
    datapoint["image"] = tf.image.rgb_to_hsv(datapoint["image"])

    mask = tf.py_function(value_augmentation_spot_light,
                          inp=[datapoint["image"].shape, aug_param["value"], aug_param["strength_spot"]], Tout=tf.float32)
    gaussian_noise = tf.random.stateless_uniform([1], seed, minval=0, maxval=aug_param["gaussian_value"])
    mask = tf.keras.layers.GaussianNoise(gaussian_noise)(mask, training=True)
    datapoint["image"] = mask + datapoint["image"]
    datapoint["image"] = tf.clip_by_value(datapoint["image"], 0.0, 1.0)

    # convert back to RGB of uint8: [0,255]
    datapoint["image"] = tf.image.hsv_to_rgb(datapoint["image"])
    datapoint["image"] = tf.image.convert_image_dtype(datapoint["image"], tf.uint8, saturate=True)

    seed = rng.make_seeds(2)[0]
    datapoint["image"] = tf.image.stateless_random_flip_left_right(datapoint["image"], seed)
    datapoint["mask"] = tf.image.stateless_random_flip_left_right(datapoint["mask"], seed)

    seed = rng.make_seeds(2)[0]
    datapoint["image"] = tf.image.stateless_random_flip_up_down(datapoint["image"], seed)
    datapoint["mask"] = tf.image.stateless_random_flip_up_down(datapoint["mask"], seed)

    return {'image': datapoint['image'], 'mask': datapoint['mask']}

def normalize_fun(datapoint, has_mask):
    datapoint['image'] = tf.cast(datapoint['image'], tf.float32)/127.5-1.0
    if has_mask:
        return {'image': datapoint['image'], 'mask': datapoint['mask']}
    else:
        return {'image': datapoint['image']}

def dataset_processing(ds, cache=False, shuffle=False, batch_size=False, augment=False, prefetch=False, rng=None,
                       aug_param=None, normalize=False, has_mask=True, img_count=0):
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(img_count, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda x: augment_mapping(x, rng, aug_param), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if normalize:
        ds = ds.map(lambda x: normalize_fun(x, has_mask), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(split_dataset_dictionary, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if batch_size:
        ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


def clean_model_directories(paths, del_old_checkpoints, del_old_tensorboard):
    if del_old_checkpoints:
        shutil.rmtree(paths['CKPT'])
        os.makedirs(paths['CKPT'])

    if del_old_tensorboard:
        shutil.rmtree(paths['TBLOGS'])
        os.makedirs(paths['TBLOGS'])
