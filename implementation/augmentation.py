from keras.preprocessing.image import ImageDataGenerator
import os
import glob
import tensorflow as tf
import shutil
from PIL import Image
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def DataGenerator(image_path, mask_path, augmentation_image, augmentation_mask, input_type, img_size=(256, 256),
                  batch_size=32, color_mode='rgb', data_format='channels_last', prepare_array=True):

    # delete last time contents and create new folder to save augmentation data
    if os.path.exists(augmentation_image):
        shutil.rmtree(augmentation_image)

    if os.path.exists(augmentation_mask):
        shutil.rmtree(augmentation_mask)

    if not os.path.exists(augmentation_image):
        os.makedirs(augmentation_image)
    if not os.path.exists(augmentation_mask):
        os.makedirs(augmentation_mask)

    data_gen_args = dict(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        rotation_range=15,
        data_format=data_format)

    val_data_gen_args = dict(
        rescale=1./255,
        data_format=data_format
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    val_img_datagen = ImageDataGenerator(**val_data_gen_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_args)

    all_images = []
    all_masks = []
    train_img = None
    train_mask = None
    val_img = None
    val_mask = None

    if color_mode == 'rgb':
        mode = 1
    elif color_mode == 'grayscale':
        mode = 0



    if prepare_array:
        for img in glob.glob(image_path + '/' + input_type + '/*.jpg'):
            name = os.path.basename(img)
            name = os.path.splitext(name)[0]

            im = cv2.imread(img, mode)
            # print(name)
            mask = cv2.imread(mask_path + '/mask/' + name + '.tif', 0)

            im = cv2.resize(im, img_size)
            # print(np.shape(im))
            mask = cv2.resize(mask, img_size)
            # print(np.shape(mask))
            image = tf.keras.preprocessing.image.img_to_array(im, data_format=data_format)
            mask = tf.keras.preprocessing.image.img_to_array(mask, data_format=data_format)
            all_images.append(image)
            all_masks.append(mask)

        all_images = np.array(all_images)
        all_masks = np.array(all_masks)
        print('image shape:', np.shape(all_images))
        print('mask shape:', np.shape(all_masks))

        train_img, val_img, train_mask, val_mask = train_test_split(all_images, all_masks, test_size=0.1)

        print(np.shape(train_img))
        print(np.shape(val_img))

    seed = 1
    image_datagen.fit(train_img, augment=True, seed=seed)
    mask_datagen.fit(train_mask, augment=True, seed=seed)

    # if from_directory:
    #     image_generator = image_datagen.flow_from_directory(
    #         image_path,
    #         target_size=img_size,
    #         color_mode=color_mode,
    #         class_mode=None,
    #         batch_size=batch_size,
    #         # save_to_dir=augmentation_image,
    #         seed=seed)
    #
    #     mask_generator = mask_datagen.flow_from_directory(
    #         mask_path,
    #         target_size=img_size,
    #         color_mode=color_mode,
    #         class_mode=None,
    #         batch_size=batch_size,
    #         # save_to_dir=augmentation_mask,
    #         seed=seed)

    if True:
        image_generator = image_datagen.flow(
            train_img,
            batch_size=batch_size,
            # save_to_dir=augmentation_mask,
            seed=seed
        )

        mask_generator = mask_datagen.flow(
            train_mask,
            batch_size=batch_size,
            # save_to_dir=augmentation_mask,
            seed=seed
        )

        val_img_generator = val_img_datagen.flow(
            val_img,
            batch_size=batch_size,
            seed=seed
        )

        val_mask_generator = val_mask_datagen.flow(
            val_mask,
            batch_size=batch_size,
            seed=seed
        )

    train_generator = zip(image_generator, mask_generator)
    val_generator = zip(val_img_generator, val_mask_generator)

    print('flow, done!')
    return train_generator, val_generator, len(image_generator), len(val_img_generator)


