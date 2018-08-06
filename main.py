from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from keras.applications import vgg19
from keras import backend as K
import os

def main(base_image_path, style_reference_image_path, result_path, iterations, content_weight, style_weight):

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(img_nrows, img_ncols))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def deprocess_image(x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((3, img_nrows, img_ncols))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((img_nrows, img_ncols, 3))
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x    

    class Evaluator(object):

        def __init__(self):
            self.loss_value = None
            self.grads_values = None

        def loss(self, x):
            assert self.loss_value is None
            loss_value, grad_values = eval_loss_and_grads(x)
            self.loss_value = loss_value
            self.grad_values = grad_values
            return self.loss_value

        def grads(self, x):
            assert self.loss_value is not None
            grad_values = np.copy(self.grad_values)
            self.loss_value = None
            self.grad_values = None
            return grad_values

    def gram_matrix(x):
        assert K.ndim(x) == 3
        if K.image_data_format() == 'channels_first':
            features = K.batch_flatten(x)
        else:
            features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram = K.dot(features, K.transpose(features))
        return gram

    def style_loss(style, combination):
        assert K.ndim(style) == 3
        assert K.ndim(combination) == 3
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = img_nrows * img_ncols
        return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

    def content_loss(base, combination):
        return K.sum(K.square(combination - base))

    def total_variation_loss(x):
        assert K.ndim(x) == 4
        if K.image_data_format() == 'channels_first':
            a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1])
            b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:])
        else:
            a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
            b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    def eval_loss_and_grads(x):
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, img_nrows, img_ncols))
        else:
            x = x.reshape((1, img_nrows, img_ncols, 3))
        outs = f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
    # parser.add_argument('base_image_path', metavar='base', type=str, help='Path to the image to transform.')
    # parser.add_argument('style_reference_image_path', metavar='ref', type=str, help='Path to the style reference image.')
    # parser.add_argument('result_prefix', metavar='res_prefix', type=str, help='Prefix for the saved results.')
    # parser.add_argument('iter', type=int, default=10, help='Number of iterations to run.')
    # parser.add_argument('--content_weight', type=float, default=1.0, required=False, help='Content weight.')
    # parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='Style weight.')
    parser.add_argument('--tv_weight', type=float, default=1.0, required=False, help='Total Variation weight.')

    args = parser.parse_args()
    style_reference_image_path = style_reference_image_path
    iterations = iterations

    total_variation_weight = args.tv_weight
    style_weight = style_weight
    content_weight = content_weight
    
    img_nrows = 32
    img_ncols = 32


    result_name = '0.png'
    base_image_path = base_image_path + result_name

    base_image = K.variable(preprocess_image(base_image_path))
    style_reference_image = K.variable(preprocess_image(style_reference_image_path))

    if K.image_data_format() == 'channels_first':
        combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
    else:
        combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

    input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)
    
    model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
    print('Model loaded.')

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    
    for j in range(60000):

        result_name = str(j) + '.png'
        base_image_path = base_image_path + result_name


        base_image = K.variable(preprocess_image(base_image_path))
        style_reference_image = K.variable(preprocess_image(style_reference_image_path))

        if K.image_data_format() == 'channels_first':
            combination_image = K.placeholder((1, 3, img_nrows, img_ncols))
        else:
            combination_image = K.placeholder((1, img_nrows, img_ncols, 3))

        input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)

        loss = K.variable(0.)
        layer_features = outputs_dict['block5_conv2']
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss += content_weight * content_loss(base_image_features, combination_features)

        feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (style_weight / len(feature_layers)) * sl
        loss += total_variation_weight * total_variation_loss(combination_image)

        grads = K.gradients(loss, combination_image)

        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        f_outputs = K.function([combination_image], outputs)
            
        x = preprocess_image(base_image_path)   
        evaluator = Evaluator()
        i = 0
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        print('Current loss value:', min_val)
        # save current generated image
        img = deprocess_image(x.copy())
        img_name = result_path + result_name
        save_img(img_name, img)
        end_time = time.time()
        print('Save as', img_name)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
        print('Total Running Time: %ds' % (end_time - begin_time))
        print()

# file_list = list()
# for i in os.listdir(path):
#         file_list.append(i)
#     if '.png' in i:

begin_time = time.time()
main('mnist/channel3_32/x_train/', 'style/halftone_32.png', 'mnist/halftone/x_train/', 1, 1.0, 1.0)
