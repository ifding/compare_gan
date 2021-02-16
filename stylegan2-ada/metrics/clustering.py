# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Inception Score (IS) from the paper
"Improved techniques for training GANs"."""

import pickle
import numpy as np
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from metrics import metric_base
from training import dataset

from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score

def compute_purity(y_pred, y_true):
        """
        Calculate the purity, a measurement of quality for the clustering
        results.

        Each cluster is assigned to the class which is most frequent in the
        cluster.  Using these classes, the percent accuracy is then calculated.

        Returns:
          A number between 0 and 1.  Poor clusterings have a purity close to 0
          while a perfect clustering has a purity of 1.
        """

        # get the set of unique cluster ids
        clusters = set(y_pred)

        # find out what class is most frequent in each cluster
        cluster_classes = {}
        correct = 0
        for cluster in clusters:
            # get the indices of rows in this cluster
            indices = np.where(y_pred == cluster)[0]

            cluster_labels = y_true[indices]
            majority_label = np.argmax(np.bincount(cluster_labels))
            correct += np.sum(cluster_labels == majority_label)

            #cor = np.sum(cluster_labels == majority_label)
            #print(cluster, len(indices), float(cor)/len(indices))

        return float(correct) / len(y_pred)



#----------------------------------------------------------------------------

class CL(metric_base.MetricBase):
    def __init__(self, num_images, num_splits, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.num_splits = num_splits
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, E, G_kwargs, num_gpus, **_kwargs): # pylint: disable=arguments-differ
        minibatch_size = num_gpus * self.minibatch_per_gpu
        dataset_obj = dataset.load_dataset(**self._dataset_args)
        dataset_obj.configure(minibatch_size)
        trues = np.empty([self.num_images, 10], dtype=np.int32)
        preds = np.empty([self.num_images, 10], dtype=np.float32)
        # Construct TensorFlow graph.
        result_expr = []
        true_labels = []
        for gpu_idx in range(num_gpus):
            with tf.device(f'/gpu:{gpu_idx}'):
                E_clone = E.clone()
                images, labels = dataset_obj.get_minibatch_tf()
                outputs = E_clone.get_output_for(images, labels, **G_kwargs)
                output_logits = outputs[:, 512:]
                output_labels = tf.nn.softmax(output_logits)
                result_expr.append(output_labels)
                true_labels.append(labels)

        # Calculate activations for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            trues[begin:end] = np.concatenate(tflib.run(true_labels), axis=0)[:end-begin]
            preds[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]

        labels_true = np.argmax(trues, axis=1)
        labels_pred = np.argmax(preds, axis=1)

        purity = compute_purity(labels_pred, labels_true)
        ari = adjusted_rand_score(labels_true, labels_pred)
        nmi = normalized_mutual_info_score(labels_true, labels_pred)

        self._report_result(purity, suffix='purity')
        self._report_result(ari, suffix='ari')
        self._report_result(nmi, suffix='nmi')


#----------------------------------------------------------------------------
