import os
import cv2
import argparse
import tensorflow as tf
import numpy as np
import ssd
import dataset
import utils

from profiles import SSD_300


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def list_files(directory_path, exts):
    from fnmatch import fnmatch

    def check_extension(file_path, exts):
        for ext in exts:
            curr_ext = os.path.splitext(file_path)[1]
            if fnmatch(curr_ext, ext):
                return True
        return False

    file_list = list()
    for root, subdirs, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            if check_extension(file_path, exts):
                file_list.append(file_path)
    return file_list


def run(input_path, output_path, checkpoint_path, batch_size=20, profile=SSD_300):
    ds = dataset.VocDataset()
    make_dir(output_path)
    tf.reset_default_graph()
    default_boxes_rel = utils.get_prior_boxes(profile)

    with tf.Session() as session:

        net = ssd.SSD(session, profile, len(ds.label_names))
        net.load_metagraph(checkpoint_path, continue_training=False)

        file_list = list_files(input_path, ["*.jpg", "*.png"])
        for file_batch in dataset.batch_iterator(file_list, batch_size):
            data_list = []
            for filepath in file_batch:
                img_raw = cv2.imread(filepath, cv2.IMREAD_COLOR)
                data = cv2.resize(img_raw, profile.imgsize).astype(np.float)
                data_list.append(data)
            data_arr = np.array(data_list, dtype=np.float32)

            feed = {net.input: data_arr}
            result = session.run(net.output, feed_dict=feed)

            detections_per_file = []
            for i in range(result.shape[0]):
                detections = utils.get_filtered_result_bboxes(result[i], default_boxes_rel, profile.imgsize)
                detections_per_file.append([file_batch[i], detections])

            for filepath, dets in detections_per_file:
                basename = os.path.basename(filepath)
                destpath = os.path.join(output_path, basename)
                utils.draw_detections(destpath, filepath, dets, label_names=ds.label_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='/data/Workspace/data/test', help='data directory')
    parser.add_argument('--dest-dir', default='./result', help='output directory')
    parser.add_argument('--checkpoint',
                        default='/data/Workspace/github/ssd.tf/checkpoints/ssd/checkpoint-epoch-009.ckpt.meta',
                        help='path to pretrained model(checkpoint file)')
    parser.add_argument('--batch-size', type=int, default=20, help='batch size')
    args = parser.parse_args()

    data_dir = args.data_dir
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    run(data_dir, "./sample-out", checkpoint, batch_size)