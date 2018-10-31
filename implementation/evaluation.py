import glob
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import cv2
import keras
from u_net import *
import compile_train as Ct


def Generate_Testset(test_img_path, test_mask_path, color_mode, input_type, img_size, data_format='channels_last'):

    if color_mode == 'rgb':
        mode = 1
    elif color_mode == 'grayscale':
        mode = 0

    all_images = []
    all_masks = []
    name_list = []

    for img in glob.glob(test_img_path + '/' + input_type + '/*.jpg'):
        name = os.path.basename(img)
        pure_name = os.path.splitext(name)[0]
        name_list.append(name)

        im = cv2.imread(img, mode)
        # print(name)
        mask = cv2.imread(test_mask_path + '/mask/' + pure_name + '.tif', 0)

        im = cv2.resize(im, img_size)
        mask = cv2.resize(mask, img_size)
        # print(np.shape(im))
        # print(np.shape(mask))
        image = tf.keras.preprocessing.image.img_to_array(im, data_format=data_format)
        mask = tf.keras.preprocessing.image.img_to_array(mask, data_format=data_format)
        all_images.append(image)
        all_masks.append(mask)

    all_images = np.array(all_images) * (1./255)
    all_masks = np.array(all_masks) * (1./255)
    print('test image shape:', np.shape(all_images))
    print('test mask shape:', np.shape(all_masks))

    return all_images, all_masks, name_list


def EvaluateAndPredict(model, x, y, img_size, test_name_list, save_path, batch_size=8):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_metric = model.evaluate(x, y, batch_size=batch_size, verbose=1)
    print('Evaluation:', loss_metric)

    results = model.predict(x)

    for i in range(len(results)):
        img = np.reshape(results[i, :, :], img_size)
        # print(np.shape(img))
        imsave(save_path + '/' + str(test_name_list[i]), img)

    return results


if __name__ == '__main__':

    width = 256
    height = 256
    first_layer_filters = 8
    model_depth = 6

    base_path = '../data/'
    train_img_path = base_path + 'train/images/'
    train_mask_path = base_path + 'train/masks/'
    test_img_path = base_path + 'test/images/'
    test_mask_path = base_path + 'test/masks/'
    prediction_path = base_path + 'prediction/'
    aug_img = base_path + 'aug_img/'
    aug_mask = base_path + 'aug_mask/'

    weight_path = base_path + 'weight/'
    weight_name = 'baseline1'

    # modify input type for different input
    input_type = 'RGB'
    data_format = 'channels_last'
    optimizer = 'Adam'
    color_mode = 'rgb'
    batch_size = 8
    lr = 0.0001
    epochs = 200

    if color_mode == 'grayscale':
        channels = 1
    elif color_mode == 'rgb' or 'rbg':
        channels = 3
    keras.backend.set_image_data_format(data_format)

    ################################################################
    model = UNET()
    u_net = model.BuildUnet((width, height, channels), first_layer_filters, model_depth, dropout=False,
                            _data_format=data_format)
    Ct.Compile(u_net, optimizer=optimizer, lr=lr, pre_weights=weight_path+weight_name+'.h5')
    print('Build model, done')

    ################################################################
    x, y, name_list = Generate_Testset(test_img_path, test_mask_path, color_mode=color_mode, img_size=(width, height),
                                       data_format=data_format)
    print('Prepare test set, done')

    EvaluateAndPredict(u_net, x, y, img_size=(width, height), test_name_list=name_list, save_path=prediction_path,
                       batch_size=8)



















