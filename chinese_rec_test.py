# coding: utf-8

import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
from PIL import Image


logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', False, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 200, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 5, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 50, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './data/train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './data/test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './logs', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 50, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS


class DataIterator:
    def __init__(self, data_dir):
        if not os.path.exists(data_dir):
            print('data dir not exists...')
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        # truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        # print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            # if root < truncate_path:
            self.image_names += [os.path.join(root, file_path) for file_path in file_list if file_path != '.DS_Store']
        random.shuffle(self.image_names)
        self.labels = np.array([self.generate_label(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names])
        # self.labels = np.array([1 for _ in self.image_names])


    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def generate_label(file_name):
        label = [0] * 2500
        l = len(file_name)
        c = l / 4
        for i in range(c):
            hit = int(file_name[i * 4: i * 4 + 4])
            label[hit] = 1
        return label

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        print '------'
        print images.get_shape()
        images = self.reshape(images, [[0, 1, 2]])
        print images.get_shape()
        print '------'
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)

        return image_batch, label_batch

    def get_shape(self, tensor):
        static_shape = tensor.get_shape().as_list()
        dynamic_shape = tf.unstack(tf.shape(tensor))
        dims = [s[1] if s[0] is None else s[0] for s in zip(static_shape, dynamic_shape)]
        return dims

    def reshape(self, tensor, dims_list):
        shape = self.get_shape(tensor)
        dims_prob = []
        for dims in dims_list:
            if isinstance(dims, int):
                dims_prob.append(shape[dims])
            elif all([isinstance(shape[d], int) for d in dims]):
                dims_prob.append(np.prod([shape[d] for d in dims]))
            else:
                dims_prob.append(tf.prod([shape[d] for d in dims]))
        tensor = tf.reshape(tensor, dims_prob)
        return tensor


# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
#     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
#     return result


# 产生随机变量，符合 normal 分布
# 传递 shape 就可以返回weight和bias的变量
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义2维的 convolutional 图层
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    # strides 就是跨多大步抽取信息
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # 定义 pooling 图层


def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    # 用pooling对付跨步大丢失信息问题
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def build_graph(top_k):
    # define placeholder for inputs to network

    images = tf.placeholder(tf.float32, [None, 4096])  # 784＝28x28
    labels = tf.placeholder(tf.float32, [None, 2500])
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(images, [-1, 64, 64, 1])  # 最后一个1表示数据是黑白的
    # print(x_image.shape)  # [n_samples, 28,28,1]

    ## 1. conv1 layer ##
    #  把x_image的厚度1加厚变成了32
    W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5, in size 1, out size 32
    b_conv1 = bias_variable([32])
    # 构建第一个convolutional层，外面再加一个非线性化的处理relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
    # 经过pooling后，长宽缩小为14x14
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32

    ## 2. conv2 layer ##
    # 把厚度32加厚变成了64
    W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in size 32, out size 64
    b_conv2 = bias_variable([64])
    # 构建第二个convolutional层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
    # 经过pooling后，长宽缩小为7x7
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64

    ## 3. func1 layer ##
    # 飞的更高变成1024
    W_fc1 = weight_variable([16 * 16 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
    # 把pooling后的结果变平
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    ## 4. func2 layer ##
    # 最后一层，输入1024，输出size 10，用 softmax 计算概率进行分类的处理
    W_fc2 = weight_variable([1024, 2500])
    b_fc2 = bias_variable([2500])
    logits = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    print 'prediction'
    print logits.get_shape()
    print 'labels'
    print labels.get_shape()

    # the error between prediction and real data
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(logits),
                                                  reduction_indices=[1]))  # loss
    print 'cross_entropy'
    print cross_entropy
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    print 'loss'
    print loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1),tf.argmax(labels, 1)), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    # accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy,
            # 'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir='scripts/data/train/')
    print train_feeder.image_names
    print train_feeder.labels
    test_feeder = DataIterator(data_dir='scripts/data/test/')
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph(top_k=1)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            while not coord.should_stop():
                print 'training ... '
                start_time = time.time()

                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.8}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                print 'loss_val'
                print loss_val
                print type(loss_val)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    logger.info('Here break!')
                    break
                if step % FLAGS.eval_steps == 1:
                    logger.info('Here evaluate!')
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0}
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    logger.info('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    logger.info('Here save!')
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)


def validation():
    print('validation')
    test_feeder = DataIterator(data_dir='../data/test/')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph(3)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0}
                batch_labels, probs, indices, acc_1 = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy']], feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1)"
                            .format(i, end_time - start_time, acc_1))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
            logger.info('top 1 accuracy {0} top k accuracy'.format(acc_top_1))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


def inference(image):
    print('inference')
    temp_image = Image.open(image).convert('L')
    temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
    temp_image = np.asarray(temp_image) / 255.0
    temp_image = temp_image.reshape([-1, 64, 64, 1])
    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image, graph['keep_prob']: 1.0})
    return predict_val, predict_index


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()

    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Write file ends')
    elif FLAGS.mode == 'inference':
        image_path = '../data/test/00190/13320.png'
        final_predict_val, final_predict_index = inference(image_path)
        logger.info('the result info label {0} predict index {1} predict_val {2}'.format(190, final_predict_index,
                                                                                         final_predict_val))

if __name__ == "__main__":
    tf.app.run()
