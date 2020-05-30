# -*- coding: utf-8 -*-
"""STL-10 画像データセット"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_STL_IMAGE_SIZE = 96
_STL_IMAGE_SHAPE = (_STL_IMAGE_SIZE, _STL_IMAGE_SIZE, 3)

_CITATION = """\
@TECHREPORT{Coates11ananalysisofsignlelayer,
    author = {Adam Coates, Honglak Lee, Andrew Y. Ng},
    title = {\
        An Analysis of Single Layer Networks in Unsupervised Feature Learning\
    },
    institution = {},
    year = {2011}
}
"""
#_STL_TRAIN_FILENAMES = [('train_X.bin', 'train_y.bin')]
_STL_TRAIN_FILENAMES = [('train_X.bin', 'train_y.bin'), ('unlabeled_X.bin', None)]
_STL_TEST_FILENAMES = [('test_X.bin', 'test_y.bin')]
#_STL_UNLABELED_FILENAMES = [('unlabeled_X.bin', None)]

_STL_LABEL_KEYS = ['label']
_STL_LABEL_NAME_FILENAMES = ['class_names.txt']


class Stl10(tfds.core.GeneratorBasedBuilder):
    """STL-10"""

    VERSION = tfds.core.Version("1.0.0")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("""\
                5000訓練画像、8000テスト画像、100000教師無し学習用画像からなる画像データセットです。\
                訓練画像とテスト画像に関しては10種類の内から1つのラベルが付与されています。
                1画像のサイズは96x96x3です。\
            """),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=_STL_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(num_classes=10),
            }),
            supervised_keys=("image", "label"),
            urls=["https://cs.stanford.edu/~acoates/stl10/"],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        stl_path = dl_manager.download_and_extract(
            'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
        )
        stl_dir = 'stl10_binary'
        stl_path = os.path.join(stl_path, stl_dir)

        for label_key, label_filename\
                in zip(_STL_LABEL_KEYS, _STL_LABEL_NAME_FILENAMES):
            label_file_path = os.path.join(stl_path, label_filename)
            with tf.io.gfile.GFile(label_file_path) as f:
                label_names = [name for name in f.read().split("\n") if name]
            self.info.features[label_key].names = label_names

        # Define the splits
        def gen_filenames(filenames):
            for data_filename, label_filename in filenames:
                if label_filename:
                    yield\
                        os.path.join(stl_path, data_filename),\
                        os.path.join(stl_path, label_filename)
                else:
                    yield os.path.join(stl_path, data_filename), None

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=100,  # S3版では無視されます。
                gen_kwargs={
                    "filepaths": gen_filenames(_STL_TRAIN_FILENAMES)
                }
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                num_shards=10,  # S3版では無視されます。
                gen_kwargs={
                    "filepaths": gen_filenames(_STL_TEST_FILENAMES)
                }
            ),
           # tfds.core.SplitGenerator(
           #     name='unlabeled',
           #     num_shards=100,  # S3版では無視されます。
           #     gen_kwargs={
           #         "filepaths": gen_filenames(_STL_UNLABELED_FILENAMES)
           #     }
           # ),
        ]

    def _generate_examples(self, filepaths):
        """STL-10のExamplesを生成します。

        Args:
        filepaths list(tuple(str, str)):
            (データのファイルパス, ラベルのファイルパス)からなるタプルを入力すると、\
            指定されたファイルパスからデータを読み出します。

        Yields:
            STL-10のExamplesが返ります。
        """
        label_keys = _STL_LABEL_KEYS
        index = 0  # データは毎回同じ順序で読み込まれるためIndexをキーとします。
        for data_filepath, label_filepath in filepaths:
            if label_filepath:
                for np_image, labels\
                        in zip(
                            _load_data(data_filepath),
                            _load_label(label_filepath, len(label_keys))):
                    record = dict(zip(label_keys, labels))
                    record["image"] = np_image
                    yield record
                    index += 1
            else:
                for np_image in _load_data(data_filepath):
                    record = dict(zip(label_keys, np.array([0])))
                    record["image"] = np_image
                    yield record
                    index += 1


def _load_data(path):
    """Yields np_image."""
    with tf.io.gfile.GFile(path, "rb") as f:
        while True:
            try:
                data = f.read(27648)
                image = np.frombuffer(
                    data,
                    dtype=np.uint8,
                    count=27648
                ).reshape(
                    (3, _STL_IMAGE_SIZE, _STL_IMAGE_SIZE)
                ).transpose((2, 1, 0))
                yield image
            except ValueError:
                break


def _load_label(path, labels_number=1):
    """Yields labels."""
    with tf.io.gfile.GFile(path, "rb") as f:
        data = f.read()
    offset = 0
    max_offset = len(data)
    while offset < max_offset:
        labels = np.frombuffer(
            data,
            dtype=np.uint8,
            count=labels_number,
            offset=offset
        ).reshape((labels_number,)) - 1
        offset += labels_number
        yield labels
