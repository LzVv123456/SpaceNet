from keras import backend as k
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import os
import keras.backend as K


def Iou_Coef(y_true, y_pred):
    #y_pred = K.round(K.clip(y_pred, 0, 1))
    #y_pred = y_pred>0.5
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return jac


def Iou_Metric(y_true, y_pred):
    y_pred = K.round(K.clip(y_pred, 0, 1))
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + K.epsilon()) / (sum_ - intersection + K.epsilon())
    return jac


def Iou_Coef_Loss(y_true, y_pred):
    return -Iou_Coef(y_true, y_pred)


def Compile(model, optimizer='SGD', lr=0.01, pre_weights=False):
    if optimizer == 'SGD':
        optimizer = SGD(lr=lr, decay=5e-4, momentum=0.99)
    if optimizer == 'Adam':
        optimizer = Adam(lr=lr, decay=5e-4)

    model.compile(optimizer=optimizer, loss=Iou_Coef_Loss, metrics=[Iou_Metric])

    if pre_weights:
        model.load_weights(pre_weights)


def Train(model, train_generator, validation_generator, weight_path, steps_per_epoch, val_step, epochs=50, name="bestTrainWeight"):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    # modelCheckpoint, save best only
    modelCheckpoint = ModelCheckpoint(weight_path + "/" + name + ".h5", monitor='val_loss', save_best_only=True,
                                      save_weights_only=True)
    model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        validation_steps=val_step,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[modelCheckpoint]
    )