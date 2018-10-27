

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#----------------------------------------------------
# MIT License
#
# Copyright (c) 2017 Rishi Rai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#----------------------------------------------------


import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import cv2
import math
import pickle
from sklearn.svm import SVC
from scipy import misc
import detect_face
from six.moves import xrange
import align_dataset_mtcnn as mtcnn

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(args.model)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
            print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)

            while 1:
                ret, frame = cap.read()
                images, bboxs = mtcnn.get_multi_bbox(frame)
                if len(images) < 1:
                    continue
                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)

                predictions = model.predict_proba(emb)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                # plot camera image
                for i, (x1, y1, x2, y2, acc) in enumerate(bboxs):
                    x1 = int(np.maximum(x1, 0))
                    y1 = int(np.maximum(y1, 0))
                    x2 = int(np.minimum(x2, frame.shape[1]))
                    y2 = int(np.minimum(y2, frame.shape[0]))

                    font = cv2.FONT_HERSHEY_PLAIN
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, class_names[best_class_indices[i]]+":{:.2f}".format(best_class_probabilities[i])
                                , (int(x1), int(y1) + 30), font, 3, (0, 0, 0), 2, False)

                cv2.imshow("face_recognize", frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    main(parse_arguments(sys.argv[1:]))
    cap.release()
    cv2.destroyAllWindows()

