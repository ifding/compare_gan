# coding=utf-8
# Copyright 2018 Google LLC & Hwalsuk Lee.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of Self-Supervised GAN with auxiliary rotation loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl import flags
from absl import logging
from compare_gan.architectures import arch_ops as ops

from compare_gan.architectures import resnet_cifar, resnet_stl

from compare_gan.gans import loss_lib
from compare_gan.gans import modular_gan
from compare_gan.gans import utils

import gin
import numpy as np
import tensorflow as tf

import tensorflow_hub as hub

FLAGS = flags.FLAGS
NUM_ROTATIONS = 4


# pylint: disable=not-callable
@gin.configurable(blacklist=["kwargs"])
class S3GAN(modular_gan.ModularGAN):
  """S3GAN which enables auxiliary heads for the modular GAN."""

  def __init__(self, self_supervision="rotation",
               rotated_batch_fraction=gin.REQUIRED,
               weight_rotation_loss_d=1.0,
               weight_rotation_loss_g=0.2,
               project_y=False,
               use_predictor=False,
               use_soft_pred=False,
               weight_class_loss=1.0,
               use_soft_labels=False,
               **kwargs):
    """Instantiates the S3GAN.

    Args:
      self_supervision: One of [rotation_gan, None].
      rotated_batch_fraction: This must be a divisor of the total batch size.
        rotations of each images on each TPU core. For GPU training #CORES is 1.
      weight_rotation_loss_d: Weight for the rotation loss for the discriminator
        on real images.
      weight_rotation_loss_g: Weight for the rotation loss for the generator
        on fake images.
      project_y: Boolean, whether an embedding layer as in variant 1) should be
        used.
      use_predictor: Boolean, whether a predictor (classifier) should be used.
      use_soft_pred: Boolean, whether soft labels should be used for the
        predicted label vectors in 1).
      weight_class_loss: weight of the (predictor) classification loss added to
        the discriminator loss.
      use_soft_labels: Boolean, if true assumes the labels passed for real
        examples are soft labels and accordingly does not transform
      **kwargs: Additional arguments passed to `ModularGAN` constructor.
    """
    super(S3GAN, self).__init__(**kwargs)
    if use_predictor and not project_y:
      raise ValueError("Using predictor requires projection.")
    assert self_supervision in {"none", "rotation"}
    self._self_supervision = self_supervision
    self._rotated_batch_fraction = rotated_batch_fraction
    self._weight_rotation_loss_d = weight_rotation_loss_d
    self._weight_rotation_loss_g = weight_rotation_loss_g

    self.beta_cycle_gen = 1
    self.beta_cycle_label = 1 

    self._project_y = project_y
    self._use_predictor = use_predictor
    self._use_soft_pred = use_soft_pred
    self._weight_class_loss = weight_class_loss

    self._use_soft_labels = use_soft_labels

    # To safe memory ModularGAN supports feeding real and fake samples
    # separately through the discriminator. S3GAN does not support this to
    # avoid additional additional complexity in create_loss().
    assert not self._deprecated_split_disc_calls, \
        "Splitting discriminator calls is not supported in S3GAN."

  @property
  def encoder(self):
    if "stl" in self._dataset.name:
      self._encoder = resnet_stl.Encoder()
    else:
      self._encoder = resnet_cifar.Encoder(self._dataset.num_classes)
    return self._encoder

  def _enc_module_fn(self, model, batch_size):
    """Module Function to create a TF Hub module spec.
    Args:
      model: `tf.estimator.ModeKeys` value.
      batch_size: batch size.
    """
    if model not in {"enc"}:
      raise ValueError("Model {} not support in module_fn()".format(model))
    placeholder_fn = tf.placeholder if batch_size is None else tf.zeros
    is_training = False
    inputs = {}
    y = None
    inputs["images"] = placeholder_fn(
        shape=[batch_size] + list(self._dataset.image_shape),
        dtype=tf.float32,
        name="images_for_encode")
    #if self.conditional:
    #  inputs["labels"] = placeholder_fn(
    #      shape=(batch_size,),
    #      dtype=tf.int32,
    #      name="labels_for_eval")
    #  y = self._get_one_hot_labels(inputs["labels"])

    logging.info("Creating module for model %s with inputs %s and y=%s",
                 model, inputs, y)
    outputs = {}
    _, outputs["prediction"], _ = self.encoder(
        inputs["images"], y=y, is_training=is_training)

    hub.add_signature(inputs=inputs, outputs=outputs)
 

  def as_module_spec(self):
    """Returns the generator network as TFHub module spec."""
    models = ["gen", "disc"]
    default_batch_size = 64
    batch_sizes = [8, 16, 32, 64]

    if "resnet" in self._architecture:
      # Only ResNet architectures support dynamic batch size.
      batch_sizes.append(None)
      default_batch_size = None
    tags_and_args = [
        (set(), {"model": "gen", "batch_size": default_batch_size})]
    for model, bs in itertools.product(models, batch_sizes):
      tags = {model, "bs{}".format(bs)}
      args = {"model": model, "batch_size": bs}
      tags_and_args.append((tags, args))
    enc_tags_and_args = [
        ([], {"model": "enc", "batch_size": default_batch_size})]
    return [hub.create_module_spec(
        self._module_fn, tags_and_args=tags_and_args,
        drop_collections=[tf.GraphKeys.MOVING_AVERAGE_VARIABLES]),
        hub.create_module_spec(
        self._enc_module_fn, tags_and_args=enc_tags_and_args)
        ]


  def discriminator_with_additonal_heads(self, x, y, is_training):
    """Discriminator architecture with additional heads.

    Possible heads built on top of feature representation of the discriminator:
    (1) Classify the image to the correct class.
    (2) Classify the rotation of the image.

    Args:
      x: An input image tensor.
      y: One-hot encoded label. Passing all zeros implies no label was passed.
      is_training: boolean, whether or not it is a training call.

    Returns:
      Tuple of 5 Tensors: (1) discriminator predictions (in [0, 1]), (2) the
      corresponding logits, (3) predictions (logits) of the rotation of x from
      the auxiliary head, (4) logits of the class prediction from the auxiliary
      head, (5) Indicator vector identifying whether y contained a label or -1.
    """
    d_probs, d_logits, x_rep = self.discriminator(
        x, y=y, is_training=is_training)
    use_sn = self.discriminator._spectral_norm  # pylint: disable=protected-access

    is_label_available = tf.cast(tf.cast(
        tf.reduce_sum(y, axis=1, keepdims=True), tf.float32) > 0.5, tf.float32)
    assert x_rep.shape.ndims == 2, x_rep.shape

    # Predict the rotation of the image.
    rotation_logits = None
    if "rotation" in self._self_supervision:
      with tf.variable_scope("discriminator_rotation", reuse=tf.AUTO_REUSE):
        rotation_logits = ops.linear(
            x_rep,
            NUM_ROTATIONS,
            scope="score_classify",
            use_sn=use_sn)
        logging.info("[Discriminator] rotation head %s -> %s",
                     x_rep.shape, rotation_logits)

    #if not self._project_y:
    return d_probs, d_logits, rotation_logits, None, is_label_available

    # Predict the class of the image.
    #aux_logits = None
    #if self._use_predictor:
    #  with tf.variable_scope("discriminator_predictor", reuse=tf.AUTO_REUSE):
    #    aux_logits = ops.linear(x_rep, y.shape[1], use_bias=True,
    #                            scope="predictor_linear", use_sn=use_sn)
    #    # Apply the projection discriminator if needed.
    #    if self._use_soft_pred:
    #      y_predicted = tf.nn.softmax(aux_logits)
    #    else:
    #      y_predicted = tf.one_hot(
    #          tf.arg_max(aux_logits, 1), aux_logits.shape[1])
    #    y = (1.0 - is_label_available) * y_predicted + is_label_available * y
    #    y = tf.stop_gradient(y)
    #    logging.info("[Discriminator] %s -> aux_logits=%s, y_predicted=%s",
    #                 aux_logits.shape, aux_logits.shape, y_predicted.shape)

    #class_embedding = self.get_class_embedding(
    #    y=y, embedding_dim=x_rep.shape[-1].value, use_sn=use_sn)
    #d_logits += tf.reduce_sum(class_embedding * x_rep, axis=1, keepdims=True)
    #
    #d_probs = tf.nn.sigmoid(d_logits)
    #return d_probs, d_logits, rotation_logits, aux_logits, is_label_available

  def get_class_embedding(self, y, embedding_dim, use_sn):
    with tf.variable_scope("discriminator_projection", reuse=tf.AUTO_REUSE):
      # We do not use ops.linear() below since it does not have an option to
      # override the initializer.
      kernel = tf.get_variable(
          "kernel", [y.shape[1], embedding_dim], tf.float32,
          initializer=tf.initializers.glorot_normal())
      if use_sn:
        kernel = ops.spectral_norm(kernel)
      embedded_y = tf.matmul(y, kernel)
      logging.info("[Discriminator] embedded_y for projection: %s",
                   embedded_y.shape)
      return embedded_y

  def merge_with_rotation_data(self, real, fake, real_labels, fake_labels,
                               num_rot_examples):
    """Returns the original data concatenated with the rotated version."""

    # Put all rotation angles in a single batch, the first batch_size are
    # the original up-right images, followed by rotated_batch_size * 3
    # rotated images with 3 different angles. For NUM_ROTATIONS=4 and
    # num_rot_examples=2 we have labels_rotated [0, 0, 1, 1, 2, 2, 3, 3].
    real_to_rot, fake_to_rot = (
        real[-num_rot_examples:], fake[-num_rot_examples:])
    real_rotated = utils.rotate_images(real_to_rot, rot90_scalars=(1, 2, 3))
    fake_rotated = utils.rotate_images(fake_to_rot, rot90_scalars=(1, 2, 3))
    all_features = tf.concat([real, real_rotated, fake, fake_rotated], 0)
    all_labels = None
    if self.conditional:
      real_rotated_labels = tf.tile(real_labels[-num_rot_examples:], [3, 1])
      fake_rotated_labels = tf.tile(fake_labels[-num_rot_examples:], [3, 1])
      all_labels = tf.concat([real_labels, real_rotated_labels,
                              fake_labels, fake_rotated_labels], 0)

    return all_features, all_labels


  def _train_generator(self, features, labels, step, optimizer, params):
    # Set the random offset tensor for operations in tpu_random.py.
    #tpu_random.set_random_offset_from_features(features)
    # create_loss will set self.g_loss.
    self.create_loss(features, labels, params=params)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(
          self.g_loss,
          var_list=self.generator.trainable_variables + self.encoder.trainable_variables,
          global_step=step)
      if self._g_use_ema:
        g_vars = self.generator.trainable_variables
        with tf.name_scope("generator_ema"):
          logging.info("Creating moving averages of weights: %s", g_vars)
          # The decay value is set to 0 if we're before the moving-average start
          # point, so that the EMA vars will be the normal vars.
          decay = self._ema_decay * tf.cast(
              tf.greater_equal(step, self._ema_start_step), tf.float32)
          ema = tf.train.ExponentialMovingAverage(decay=decay)
          with tf.control_dependencies([train_op]):
            train_op = ema.apply(g_vars)
      with tf.control_dependencies([train_op]):
        return tf.identity(self.g_loss)


  def mmd_penalty(self, sample_qz, sample_pz):
    verbose = False
    sigma2_p = 1.0 ** 2
    kernel = 'RBF' #'IMQ'
    n = sample_qz.shape.as_list()[0]
    n = tf.cast(n, tf.int32)
    nf = tf.cast(n, tf.float32)
    half_size = (n * n - n) // 2

    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    if kernel == 'RBF':
      # Median heuristic for the sigma^2 of Gaussian kernel
      sigma2_k = tf.nn.top_k(
          tf.reshape(distances, [-1]), half_size).values[half_size - 1]
      sigma2_k += tf.nn.top_k(
          tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
      # Maximal heuristic for the sigma^2 of Gaussian kernel
      # sigma2_k = tf.nn.top_k(tf.reshape(distances_qz, [-1]), 1).values[0]
      # sigma2_k += tf.nn.top_k(tf.reshape(distances, [-1]), 1).values[0]
      # sigma2_k = opts['latent_space_dim'] * sigma2_p
      if verbose:
        sigma2_k = tf.Print(sigma2_k, [sigma2_k], 'Kernel width:')
      res1 = tf.exp( - distances_qz / 2. / sigma2_k)
      res1 += tf.exp( - distances_pz / 2. / sigma2_k)
      res1 = tf.multiply(res1, 1. - tf.eye(n))
      res1 = tf.reduce_sum(res1) / (nf * nf - nf)
      res2 = tf.exp( - distances / 2. / sigma2_k)
      res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
      stat = res1 - res2
    elif kernel == 'IMQ':
      # k(x, y) = C / (C + ||x - y||^2)
      # C = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
      # C += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]
      
      Cbase = 2. * self.z_dim * sigma2_p
      
      #if opts['pz'] == 'normal':
      #    Cbase = 2. * opts['zdim'] * sigma2_p
      #elif opts['pz'] == 'sphere':
      #    Cbase = 2.
      #elif opts['pz'] == 'uniform':
          # E ||x - y||^2 = E[sum (xi - yi)^2]
          #               = zdim E[(xi - yi)^2]
          #               = const * zdim
      #    Cbase = opts['zdim']
      stat = 0.
      for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz)
        res1 += C / (C + distances_pz)
        res1 = tf.multiply(res1, 1. - tf.eye(n))
        res1 = tf.reduce_sum(res1) / (nf * nf - nf)
        res2 = C / (C + distances)
        res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
        stat += res1 - res2
    return stat 


  def create_loss(self, features, labels, params, is_training=True):
    """Build the loss tensors for discriminator and generator.

    This method will set self.d_loss and self.g_loss.

    Args:
      features: Optional dictionary with inputs to the model ("images" should
          contain the real images and "z" the noise for the generator).
      labels: Tensor will labels. These are class indices. Use
          self._get_one_hot_labels(labels) to get a one hot encoded tensor.
      params: Dictionary with hyperparameters passed to TPUEstimator.
          Additional TPUEstimator will set 3 keys: `batch_size`, `use_tpu`,
          `tpu_context`. `batch_size` is the batch size for this core.
      is_training: If True build the model in training mode. If False build the
          model for inference mode (e.g. use trained averages for batch norm).

    Raises:
      ValueError: If set of meta/hyper parameters is not supported.
    """
    real_images = features["images"]
    if self.conditional:
      #if self._use_soft_labels:
      #  assert labels.shape[1] == self._dataset.num_classes, \
      #      ("Need soft labels of dimension {} but got dimension {}".format(
      #          self._dataset.num_classes, labels.shape[1]))
      #  real_labels = labels
      #else:
      #  real_labels = self._get_one_hot_labels(labels)
      real_labels = self._get_one_hot_labels(tf.zeros_like(labels,dtype=tf.int32))
    fake_labels = self._get_one_hot_labels(features["sampled_labels"])
    if self._experimental_joint_gen_for_disc:
      assert "generated" in features
      fake_images = features["generated"]
    else:
      logging.warning("Computing fake images for every sub step separately.")
      fake_images = self.generator(
          features["z"], y=fake_labels, is_training=is_training)

    #print(real_images)
    #print(fake_images)
    #print(features["sampled_labels"])
    #exit()

    bs = real_images.shape[0].value
    if self._self_supervision:
      assert bs % self._rotated_batch_fraction == 0, (
          "Rotated batch fraction is invalid: %d doesn't divide %d" %
          self._rotated_batch_fraction, bs)
      rotated_bs = bs // self._rotated_batch_fraction
      num_rot_examples = rotated_bs // NUM_ROTATIONS
      logging.info("bs=%s, rotated_bs=%s, num_rot_examples=%s", bs, rotated_bs,
                   num_rot_examples)
      assert num_rot_examples > 0

    # Append the data obtained by rotating the last 'num_rotated_samples'
    # from the true and the fake data.
    if self._self_supervision == "rotation":
      assert num_rot_examples <= bs, (num_rot_examples, bs)
      all_features, all_labels = self.merge_with_rotation_data(
          real_images, fake_images, real_labels, fake_labels, num_rot_examples)
    else:
      all_features = tf.concat([real_images, fake_images], 0)
      all_labels = None
      if self.conditional:
        all_labels = tf.concat([real_labels, fake_labels], axis=0)

    d_predictions, d_logits, rot_logits, aux_logits, is_label_available = (
        self.discriminator_with_additonal_heads(
            x=all_features, y=all_labels, is_training=is_training))

    z_enc_gen, z_enc_label, z_enc_logits = self.encoder(x=all_features, 
            y=None, is_training=is_training)
    logging.info("[Encoder] encoding images: %s", all_features)

    expected_batch_size = 2 * bs
    if self._self_supervision == "rotation":
      expected_batch_size += 2 * (NUM_ROTATIONS - 1) * num_rot_examples

    if d_logits.shape[0].value != expected_batch_size:
      raise ValueError("Batch size unexpected: got %r expected %r" % (
          d_logits.shape[0].value, expected_batch_size))

    prob_real, prob_fake = tf.split(d_predictions, 2)
    prob_real, prob_fake = prob_real[:bs], prob_fake[:bs]

    logits_real, logits_fake = tf.split(d_logits, 2)
    logits_real, logits_fake = logits_real[:bs], logits_fake[:bs]

    z_enc_gen_real, z_enc_gen_fake = tf.split(z_enc_gen, 2)
    z_enc_gen_real = z_enc_gen_real[:bs]
    z_enc_gen_fake = z_enc_gen_fake[:bs]

    z_enc_label_real, z_enc_label_fake = tf.split(z_enc_label, 2)
    z_enc_label_real = z_enc_label_real[:bs]
    z_enc_label_fake = z_enc_label_fake[:bs]

    z_enc_logits_real, z_enc_logits_fake = tf.split(z_enc_logits, 2)
    z_enc_logits_real = z_enc_logits_real[:bs]
    z_enc_logits_fake = z_enc_logits_fake[:bs]

    all_new_z = tf.concat([z_enc_gen_real, z_enc_gen_real], 0)
    all_new_labels = tf.concat([fake_labels, z_enc_label_real], axis=0)

    all_gen_images = self.generator(
        all_new_z, y=all_new_labels, is_training=is_training)

    new_fake_images, recon_images = tf.split(all_gen_images, 2)

    z_enc_gen_new, z_enc_label_new, z_enc_logits_new = self.encoder(
            x=new_fake_images, y=None, is_training=is_training) 
    
    # Get the true/fake GAN loss.
    self.d_loss, _, _, self.g_loss = loss_lib.get_losses(
        d_real=prob_real, d_fake=prob_fake,
        d_real_logits=logits_real, d_fake_logits=logits_fake)

    self.g_loss += tf.reduce_mean(tf.square(real_images - recon_images))

    self.g_loss += self.beta_cycle_gen * tf.reduce_mean(
            tf.square(z_enc_gen_new - z_enc_gen_real))        

    self.g_loss += self.beta_cycle_gen * self.mmd_penalty(z_enc_gen_real, features["z"])

    self.g_loss += self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=z_enc_logits_fake, 
                labels=fake_labels))
 
    self.g_loss += self.beta_cycle_label * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=z_enc_logits_new, 
                labels=fake_labels))
    
    #print(self.conditional)
    #exit()
    #print(self._self_supervision)
    #print(self._rotated_batch_fraction)
    #print(self._weight_rotation_loss_d)
    #print(self._weight_rotation_loss_g)

    #print(self._project_y)
    #print(self._use_predictor)
    #print(self._use_soft_pred)
    #print(self._weight_class_loss)

    #print(self._use_soft_labels)
    #exit()


    # At this point we have the classic GAN loss with possible regularization.
    # We now add the rotation loss and summaries if required.
    if self._self_supervision == "rotation":
      # Extract logits for the rotation task.
      rot_real_logits, rot_fake_logits = tf.split(rot_logits, 2)
      rot_real_logits = rot_real_logits[-rotated_bs:]
      rot_fake_logits = rot_fake_logits[-rotated_bs:]
      labels_rotated = tf.constant(np.repeat(
          np.arange(NUM_ROTATIONS, dtype=np.int32), num_rot_examples))
      rot_onehot = tf.one_hot(labels_rotated, NUM_ROTATIONS)
      rot_real_logp = tf.log(tf.nn.softmax(rot_real_logits) + 1e-10)
      rot_fake_logp = tf.log(tf.nn.softmax(rot_fake_logits) + 1e-10)
      real_loss = -tf.reduce_mean(tf.reduce_sum(rot_onehot * rot_real_logp, 1))
      fake_loss = -tf.reduce_mean(tf.reduce_sum(rot_onehot * rot_fake_logp, 1))
      self.d_loss += real_loss * self._weight_rotation_loss_d
      self.g_loss += fake_loss * self._weight_rotation_loss_g

      rot_real_labels = tf.one_hot(
          tf.arg_max(rot_real_logits, 1), NUM_ROTATIONS)
      rot_fake_labels = tf.one_hot(
          tf.arg_max(rot_fake_logits, 1), NUM_ROTATIONS)
      accuracy_real = tf.metrics.accuracy(rot_onehot, rot_real_labels)
      accuracy_fake = tf.metrics.accuracy(rot_onehot, rot_fake_labels)

      self._tpu_summary.scalar("loss/real_loss", real_loss)
      self._tpu_summary.scalar("loss/fake_loss", fake_loss)
      self._tpu_summary.scalar("accuracy/real", accuracy_real)
      self._tpu_summary.scalar("accuracy/fake", accuracy_fake)

    # Training the predictor on the features of real data and real labels.
    #if self._use_predictor:
    #  real_aux_logits, _ = tf.split(aux_logits, 2)
    #  real_aux_logits = real_aux_logits[:bs]

    #  is_label_available, _ = tf.split(is_label_available, 2)
    #  is_label_available = tf.squeeze(is_label_available[:bs])

    #  class_loss_real = tf.losses.softmax_cross_entropy(
    #      real_labels, real_aux_logits, weights=is_label_available)

    #  # Add the loss to the discriminator
    #  self.d_loss += self._weight_class_loss * class_loss_real
    #  self._tpu_summary.scalar("loss/class_loss_real", class_loss_real)
    #  self._tpu_summary.scalar("label_frac", tf.reduce_mean(is_label_available))
# coding=utf-8
