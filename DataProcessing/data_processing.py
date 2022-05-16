import tensorflow as tf
import os
import os.path as osp
from glob import glob
import shutil


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
        'MODEL LOADED': osp.join(base_path_model, data_model_loaded, model_loaded) if type(model_loaded) == str else None,
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


def parse_image(img_path, has_mask):

    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.uint8)

    if has_mask:
        mask_path = tf.strings.regex_replace(img_path, "images", "class_annotation")
        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)

        # For Ground Truth Check, the images are save with values * 80
        mask = tf.where(mask == tf.constant(80, tf.uint8), tf.constant(1, tf.uint8), mask)
        mask = tf.where(mask == tf.constant(160, tf.uint8), tf.constant(2, tf.uint8), mask)
        mask = tf.where(mask == tf.constant(240, tf.uint8), tf.constant(3, tf.uint8), mask)
        mask = tf.cast(mask, dtype=tf.uint8)

        return {'image': image, 'mask': mask}
    else:
        return {'image': image}


def preprocess(datapoint, height, width, half, has_mask):
    # Preprocessing Layer already added into model Pipeline: model expects input of 0-255, 3 channels
    # datapoint['image'] = tf.cast(datapoint['image'], tf.float32) / 127.5 - 1
    datapoint['image'] = tf.image.resize(datapoint['image'], (height, width))

    if has_mask:
        if half:
            mask_size = tf.convert_to_tensor([int(height / 2), int(width / 2)])
        else:
            mask_size = tf.convert_to_tensor([int(height), int(width)])

        datapoint['mask'] = tf.image.resize(datapoint['mask'], mask_size, method='nearest')

        return {'image': datapoint['image'], 'mask': datapoint['mask']}

    else:
        return {'image': datapoint['image']}


def add_noise(datapoint, noise_std, has_mask):
    datapoint['image'] = tf.keras.layers.GaussianNoise(noise_std)(datapoint['image'], training=True)

    if has_mask:
        return {'image': datapoint['image'], 'mask': datapoint['mask']}
    else:
        return {'image': datapoint['image']}


def split_dataset_dictionary(datapoint):
    if len(datapoint.keys()) == 2:
        return datapoint['image'], datapoint['mask']
    else:
        return datapoint['image']


def load_dataset(paths, dataset_name, height, width, half, max_img, has_mask=True, noise_std=None):
    files = sorted(glob(osp.join(paths["IMAGE"][dataset_name], "*.png")))
    files_filtered = [x for x in files if int(x[-8:-4]) <= max_img - 1]
    dataset = tf.data.Dataset.from_tensor_slices(files_filtered)
    dataset = dataset.map(lambda x: parse_image(x, has_mask), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: preprocess(x, height, width, half, has_mask),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if noise_std:
        dataset = dataset.map(lambda x: add_noise(x, noise_std, has_mask), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    image_count = len(files_filtered)

    print("The {mode} Dataset contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=dataset_name))

    return dataset, image_count


def augment_mapping(datapoint):
    img = datapoint["image"]

    # TODO: what should be used as seed ?

    rng = tf.random.Generator.from_seed(123, alg='philox')
    seed = rng.make_seeds(2)[0]
    print(seed)
    seed = rng.make_seeds(2)[0]
    print(seed)
    seed = rng.make_seeds(2)[0]
    print(seed)

    img = tf.image.stateless_random_brightness(img, 5, seed)
    img = tf.image.stateless_random_contrast(img, 0.5, 1.5, seed)
    img = tf.image.stateless_random_hue(img, 0.1, seed)

    datapoint["image"] = img
    return {'image': datapoint['image'], 'mask': datapoint['mask']}


def dataset_processing(ds, cache=False, shuffle=False, batch_size=False, augment=False, prefetch=False, img_count=0):
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(img_count, reshuffle_each_iteration=True)
    if batch_size:
        ds = ds.batch(batch_size)
    if augment:
        ds = ds.map(augment_mapping, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.map(split_dataset_dictionary, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
