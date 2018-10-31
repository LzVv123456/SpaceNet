import glob
import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import cv2
from skimage import measure
import keras
from u_net import *
import compile_train as Ct
import matplotlib.pyplot as plt


def Generate_Testset(test_img_path, test_mask_path, color_mode, img_size, input_type, data_format='channels_last'):

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
        name_list.append(pure_name)

        im = cv2.imread(img, mode)
        # print(name)
        mask = np.load(test_mask_path + '/distance/' + pure_name + '.npy')

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

    # loss_metric = model.evaluate(x, y, batch_size=batch_size, verbose=1)
    # print('Evaluation:', loss_metric)

    results = model.predict(x)
    print(np.shape(results))

    for i in range(len(results)):
        img = np.reshape(results[i, :, :, :], img_size)
        # print(np.shape(img))
        print(np.amax(img), np.amin(img))
        new_img = img.copy() * 255

        # img = cv2.imread(save_path + '/' + str(test_name_list[i]) + '.png')
        # Z = img.reshape((-1, 3))
        # # convert to np.float32
        # Z = np.float32(Z)
        #
        # # define criteria, number of clusters(K) and apply kmeans()
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # K = 10
        # ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        #
        # # Now convert back into uint8, and make original image
        # center = np.uint8(center)
        # res = center[label.flatten()]
        # res2 = res.reshape((img.shape))
        # cv2.imwrite('/Users/zhuoweili/Desktop/test/' + str(test_name_list[i]) + '.png', res2)


        # img = cv2.imread(save_path + '/' + str(test_name_list[i]) + '.png')
        # edges = cv2.Canny(img, 100, 200)
        # plt.subplot(121), plt.imshow(img, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.subplot(122), plt.imshow(edges, cmap='gray')
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.show()

        cv2.imwrite(save_path + '/' + str(test_name_list[i]) + '.jpg', new_img)
        new_img = cv2.imread(save_path + '/' + str(test_name_list[i]) + '.jpg', 0)
        contours = measure.find_contours(new_img, 235, 'low', 'low')
        # Display the image and plot all contours found
        fig, ax = plt.subplots()
        ax.imshow(new_img, interpolation='nearest', cmap=plt.cm.gray)

        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(save_path + '/' + str(test_name_list[i]) + '.png', bbox_inches='tight')

    return results


if __name__ == '__main__':

    # parameters and hyper-parameters
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
    weight_name = '2-baseline3'

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
    x, y, name_list = Generate_Testset(test_img_path, test_mask_path, color_mode=color_mode, input_type=input_type,
                                       img_size=(width, height), data_format=data_format)
    print('Prepare test set, done')

    EvaluateAndPredict(u_net, x, y, img_size=(width, height), test_name_list=name_list, save_path=prediction_path,
                       batch_size=8)



















