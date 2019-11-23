# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.append('../../')
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from SSD_mobilenet_v1.tool.config import Config
from SSD_mobilenet_v1.tool.get_anchors import get_Anchors
from SSD_mobilenet_v1.tool.ssd_predict import predict
from datetime import datetime
from SSD_mobilenet_v1.tool import mobilenet_v1
from sklearn.externals import joblib


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = tf.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = tf.concat((yx1, yx2), axis=-1)
    return bboxes


def new_conv2d(net, channel, stride, index):
    net = slim.conv2d(net, channel / 2, [1, 1],
                      scope='MobilenetV1/Conv2d_13_pointwise_1_Conv2d_%d_1x1_%d' % (index, channel / 2))
    if stride == 2:
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
    net = slim.conv2d(net, channel, [3, 3], stride=stride, padding='VALID',
                      scope='MobilenetV1/Conv2d_13_pointwise_2_Conv2d_%d_3x3_s2_%d' % (index, channel))
    return net
    pass


class SSD():
    def __init__(self, config):
        self.config = config
        print(self.config.Mean)
        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.anchors = get_Anchors(config.img_size, config.s_min, config.s_max, config.num_anchors, config.map_size,
                                   config.stride_size, )

        self.anchors = tf.constant(self.anchors)

    def handle_im(self, im_input, size):
        im = tf.image.decode_jpeg(im_input, channels=3)

        h = tf.shape(im)[0]
        w = tf.shape(im)[1]
        im = tf.image.resize_images(im, [size, size], method=0)
        im = tf.to_float(im)
        h = tf.to_float(h)
        w = tf.to_float(w)
        sh = h / size
        sw = w / size
        box_scale = tf.concat([[sh], [sw], [sh], [sw]], axis=0)
        im = im[None]

        return im, box_scale
        pass

    def build_net(self, ):
        self.im_input = tf.placeholder(tf.string, name='input')
        im, b_scale, = self.handle_im(self.im_input, size=self.config.img_size)
        im.set_shape(tf.TensorShape([None, self.config.img_size, self.config.img_size, 3]))
        im = im / 255 * 2 - 1

        batch_m = tf.shape(im)[0]
        with tf.variable_scope('FeatureExtractor'):
            with slim.arg_scope(
                    mobilenet_v1.mobilenet_v1_arg_scope(is_training=self.config.is_train, batch_norm_decay=0.9)):
                logits, end_points = mobilenet_v1.mobilenet_v1(im, min_depth=16, is_training=self.config.is_train)
                var_pre = tf.global_variables()[1:]
                net11 = end_points['Conv2d_11_pointwise']
                net13 = end_points['Conv2d_13_pointwise']
                net14 = new_conv2d(net13, 512, 2, 2)
                net15 = new_conv2d(net14, 256, 2, 3)
                net16 = new_conv2d(net15, 256, 2, 4)
                net17 = new_conv2d(net16, 128, 2, 5)
        weights_init = tf.truncated_normal_initializer(stddev=0.03)
        net = [net11, net13, net14, net15, net16, net17]
        with slim.arg_scope([slim.conv2d], activation_fn=None, weights_initializer=weights_init,
                            weights_regularizer=slim.l2_regularizer(self.config.weight_decay), normalizer_fn=None):
            for i in range(len(self.config.num_anchors)):
                with  tf.variable_scope('BoxPredictor_%d' % i):
                    with tf.variable_scope('ClassPredictor'):
                        net_cls = slim.conv2d(net[i], self.config.num_anchors[i] * (self.config.num_cls + 1), [1, 1])
                    with tf.variable_scope('BoxEncodingPredictor'):
                        net_box = slim.conv2d(net[i], self.config.num_anchors[i] * 4, [1, 1], )
                    net_cls = tf.reshape(net_cls, (batch_m, -1, self.config.num_cls + 1), )
                    net_box = tf.reshape(net_box, (batch_m, -1, 4), )
                    net[i] = tf.concat([net_cls, net_box], axis=-1)
        net = tf.concat(net, axis=1, name='net')

        print(net)
        pre_bboxes = loc2bbox(net[..., -4:] * tf.constant([0.1, 0.1, 0.2, 0.2], dtype=tf.float32), self.anchors)
        pre = predict(pre_bboxes[0], net[0][..., :-4], size=self.config.img_size, c_thresh=1e-2,
                      num_cls=self.config.num_cls)
        self.result = tf.concat([pre[..., :4] * b_scale, pre[..., 4:]], axis=-1, name='out')
        print(self.result)

    def test(self):

        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d], activation_fn=tf.nn.relu6):
            self.build_net()

        file = self.config.pre_model
        saver = tf.train.Saver()
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, file)
            Res = {}
            i = 0
            m = 10000
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res = sess.run(self.result, feed_dict={self.im_input: img})

                res = res[:, [1, 0, 3, 2, 4, 5]]

                Res[name] = res[:100]
            print(datetime.now() - time_start)
            joblib.dump(Res, 'SSD300.pkl')

    def save_pb(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
            self.build_net()

        file = '/home/zhai/PycharmProjects/Demo35/SSD/train/models/SSD300_2x.ckpt-120000'
        # file='/home/zhai/PycharmProjects/Demo35/myDNN/SSD_tf/train/models/SSD300_4.ckpt-60000'
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, file)
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['out'])
            with tf.gfile.FastGFile('model.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())

    def test_pb(self):
        pb_file = 'model.pb'
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)
        sess = tf.Session()
        with gfile.FastGFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 导入计算图
            constant_values = {}
            constant_ops = [op for op in sess.graph.get_operations() if op.type == "Const"]
            for constant_op in constant_ops:
                print(constant_op.name, )
            # 需要有一个初始化的过程
            #     sess.run(tf.global_variables_initializer())
            input_x = sess.graph.get_tensor_by_name('input:0')
            out = sess.graph.get_tensor_by_name('out:0')

            Res = {}
            i = 0
            m = 100000
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res = sess.run(out, feed_dict={input_x: img})
                res = res[:, [1, 0, 3, 2, 4, 5]]

                Res[name] = res[:100]
            print(datetime.now() - time_start)
            joblib.dump(Res, 'SSD300_2x.pkl')


if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)

    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'voc_07.tf', path + 'voc_12.tf']

    pre_model = '../train/models/SSD300_1.ckpt-90000'
    pre_model = '/home/zhai/PycharmProjects/Demo35/SSD_mobilenet_v1/train/models/SSD300_1.ckpt-60000'
    config = Config(False, Mean, files, pre_model, s_min=0.2, s_max=0.95, img_size=300, batch_size_per_GPU=16, gpus=1,
                    weight_decay=0.0005,
                    jitter_ratio=[0.3, 0.5, 0.7], crop_iou=0.45, keep_ratio=0.2, img_scale_size=[212, 150, 106, 75],
                    num_anchors=[3, 6, 6, 6, 6, 6],
                    stride_size=[16, 32, 64, 100, 150, 300], map_size=[19, 10, 5, 3, 2, 1])
    ssd = SSD(config)
    ssd.test()
    # ssd.save_pb()
    # ssd.test_pb()
    pass
