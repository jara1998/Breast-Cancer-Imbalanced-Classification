import tensorflow as tf
import cv2, os, json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train FGVC Network")

    parser.add_argument(
        "--input_path",
        help="input train/test splitting files",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--cls_names_path",
        help="path for class name file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_path",
        help="save path for converted file",
        type=str,
        required=False,
        default="."
    )
    args = parser.parse_args()
    return args


def read_and_decode(filename_queue):
    """Parses a single tf.Example into image and label tensors."""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features["image"], tf.uint8)
    image.set_shape([3 * 32 * 32])
    label = tf.cast(features["label"], tf.int32)
    return image, label


def convert_from_tfrecords(data_root, dir_name, num_class, mode, output_path, json_file_prefix, cls_names):
    if mode == 'test':
        tfrecord_path = os.path.join(data_root, dir_name, 'eval.tfrecords')
    else:
        tfrecord_path = os.path.join(data_root, dir_name, 'train.tfrecords')
    filename_queue = tf.train.string_input_producer([tfrecord_path], shuffle=False, num_epochs=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    image, label = read_and_decode(filename_queue)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    annotations = []
    try:
        step = 0
        while not coord.should_stop():
            images, labels = sess.run([image, label])
            if int(labels) % 2 != 0:
                continue
            images = cv2.cvtColor(images.reshape(3, 32, 32).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            im_path = os.path.join(output_path, json_file_prefix, 'images', cls_names[int(labels)])
            if not os.path.exists(im_path):
                os.makedirs(im_path)
            save_path = os.path.join(im_path, '{}_{}.jpg'.format(mode, step))
            cv2.imwrite(save_path, images)
            annotations.append(
                {'fpath': save_path, 'image_id': step, 'category_id': int(labels), 'category': cls_names[int(labels)]})
            step += 1
    except tf.errors.OutOfRangeError:
        print('done')
    finally:
        coord.request_stop()

    with open(os.path.join(output_path, json_file_prefix, json_file_prefix + '_{}.json'.format(mode)), 'w') as f:
        json.dump({'annotations': annotations, 'num_classes': num_class}, f)

    print('Json has been saved to',
          os.path.join(output_path, json_file_prefix, json_file_prefix + '_{}.json'.format(mode)))


if __name__ == '__main__':
    modes = ['train', 'test']
    args = parse_args()
    with open(args.cls_names_path) as f:
        cls_names = json.load(f)

    # you can add other datasets as follows
    cifar100_im_1 = {'dir': 'cifar-100-data', 'json': 'cifar50', 'class': 50}
    cifar100_im_01 = {'dir': 'cifar-100-data-im-0.1', 'json': 'cifar50_imbalance_0.1', 'class': 50}
    cifar100_im_002 = {'dir': 'cifar-100-data-im-0.02', 'json': 'cifar50_imbalance_0.02', 'class': 50}
    cifar100_im_0005 = {'dir': 'cifar-100-data-im-0.005', 'json': 'cifar50_imbalance_0.005', 'class': 50}

    for m in modes:
        convert_from_tfrecords(
            args.input_path, cifar100_im_1['dir'],
            cifar100_im_1['class'], m, args.output_path,
            cifar100_im_1['json'], cls_names
        )
        convert_from_tfrecords(
            args.input_path, cifar100_im_01['dir'],
            cifar100_im_01['class'], m, args.output_path,
            cifar100_im_01['json'], cls_names
        )
        convert_from_tfrecords(
            args.input_path, cifar100_im_002['dir'],
            cifar100_im_002['class'], m, args.output_path,
            cifar100_im_002['json'], cls_names
        )
        convert_from_tfrecords(
            args.input_path, cifar100_im_0005['dir'],
            cifar100_im_0005['class'], m, args.output_path,
            cifar100_im_0005['json'], cls_names
        )
