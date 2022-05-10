import tensorflow as tf
import os
import os.path as osp
from glob import glob
import shutil


def path_definitions(half, model, data, train, test, test_hard=None, img_only=None, model_loaded=None):
    base_path_model = '/home/david/SemesterProject/Models'
    base_path_data = '/home/david/SemesterProject/Datasets'
    paths = {
        'MODEL': osp.join(base_path_model, model),
        'CKPT': osp.join(base_path_model, model, 'CKPT'),
        'TBLOGS': osp.join(base_path_model, model, 'logs'),
        'TFLITE': osp.join(base_path_model, model, 'TFLITE'),
        'FIGURES': osp.join(base_path_model, model, 'FIGURES'),
        'MODEL LOADED': osp.join(base_path_model, model_loaded) if type(model_loaded) == str else None,
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

        return datapoint['image'], datapoint['mask']

    else:
        return datapoint['image']


def loader(paths, dataset_name, height, width, half, max_img, has_mask=True):
    files = sorted(glob(osp.join(paths["IMAGE"][dataset_name], "*.png")))
    files_filtered = [x for x in files if int(x[-8:-4]) <= max_img - 1]
    dataset = tf.data.Dataset.from_tensor_slices(files_filtered)
    dataset = dataset.map(lambda x: parse_image(x, has_mask), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: preprocess(x, height, width, half, has_mask),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # image_count = len(glob(osp.join(paths["IMAGE"][mode], "*.png")))
    image_count = len(files_filtered)

    print("The {mode} Dataset contains {IMAGES_SIZE} images.".format(IMAGES_SIZE=image_count, mode=dataset_name))

    return dataset, image_count


def dataset_processing(ds, cache=False, shuffle=False, batch_size=False, prefetch=False, img_count=0):
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(img_count, reshuffle_each_iteration=True)
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
