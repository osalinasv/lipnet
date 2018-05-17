from keras import backend as k
from keras.layers.core import Lambda


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    y_pred = y_pred[:, :, :]
    return k.ctc_batch_cost(labels, y_pred, input_length, label_length)


def CTC(name, args):
    return Lambda(ctc_lambda_func, output_shape=(1,), name=name)(args)
