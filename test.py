import argparse

import cv2
import neuralgym as ng
import numpy as np
import tensorflow as tf

from inpaint_model_gc import InpaintGCModel

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

########################################################################################################################
# Mask Generation - Change values to generate different masks
########################################################################################################################
def npmask(h, w):
    mask = np.zeros((h, w))
    num_v = 5
    for i in range(num_v):
        start_x = h//2
        start_y = w//4
        for j in range(5):
            angle = 1.5
            if i % 2 == 0:
                angle = 2 * 3.1415926 - angle
            length = 100
            brush_w = 10
            end_x = (start_x + length * np.sin(angle)).astype(np.int32)
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)

            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            start_x, start_y = end_x, end_y
    return mask.reshape(mask.shape + (1,)).astype(np.float32)

if __name__ == "__main__":
    config = ng.Config('inpaint.yml')
    if config.GPU_ID != -1:
        ng.set_gpus(config.GPU_ID)
    else:
        ng.get_gpus(config.NUM_GPUS)
    args = parser.parse_args()

    model = InpaintGCModel()
    image = cv2.imread(args.image)
    h, w, _ = image.shape
    mask = npmask(h, w)
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(image, dtype=tf.float32)
        input_mask = tf.constant(mask, dtype=tf.float32)
        output = model.build_server_graph(input_image, input_mask)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        cv2.imwrite(args.output, result[0][:, :, ::-1])
