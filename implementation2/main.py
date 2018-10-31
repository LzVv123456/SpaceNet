import os
from u_net import *
import augmentation as Aug
import compile_train as Ct
# import evaluation as Ev

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

###################################################################
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
####################################################################
# data augmentation
train_generator, validation_generator, len_train, len_val = \
    Aug.DataGenerator(train_img_path, train_mask_path, aug_img, aug_mask, input_type=input_type, img_size=(width, height),
                      batch_size=batch_size, color_mode=color_mode, data_format=data_format, prepare_array=True)

print('Augmentation, done')
##################################################################
# build and compile
model = UNET()
u_net = model.BuildUnet((width, height, channels), first_layer_filters, model_depth, dropout=False, _data_format=data_format)
Ct.Compile(u_net, optimizer=optimizer, lr=lr, pre_weights=False)

print('Build model, done')
##################################################################
# train the model
Ct.Train(u_net, train_generator, validation_generator, weight_path, steps_per_epoch=len_train, val_step=len_val,
         epochs=epochs, name=weight_name)

print('Train, done')

##################################################################
# evaluate and predict
# model = UNET()
# u_net = model.BuildUnet((width, height, channels), first_layer_filters, model_depth, dropout=False,
#                         _data_format=data_format)
# Ct.Compile(u_net, optimizer=optimizer, lr=lr, pre_weights=weight_path + weight_name + '.h5')
# print('Build model, done')
#
# x, y, name_list = Ev.Generate_Testset(test_img_path, test_mask_path, color_mode=color_mode, img_size=(width, height),
#                                       data_format=data_format)
# print('Prepare test set, done')
#
# Ev.EvaluateAndPredict(u_net, x, y, img_size=(width, height), test_name_list=name_list, save_path=prediction_path,
#                       batch_size=8)
