import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

class TensorIterator(object):

    def __init__(self, dataset, model, session, config):

        self.dataset = dataset
        self.model = model
        self.session = session
        self.config = config

        self.handle_placeholder = tf.placeholder(tf.string, shape=[])

    def create_dataset_iterator(self, mode='train'):

        dataset = tf.data.Dataset.from_tensor_slices((self.model.x, self.model.y))

        # TODO: Add shuffling
        dataset = dataset.batch(self.config.batch_size()).repeat()

        dataset_iterator = dataset.make_initializable_iterator()

        generic_iterator = tf.data.Iterator.from_string_handle(self.handle_placeholder, dataset.output_types,
                                                               dataset.output_shapes, dataset.output_classes)

        dataset_handle = self.session.run(dataset_iterator.string_handle())

        x, y = generic_iterator.get_next()
        inputs = {
            'x': x,
            'y': y
        }

        if mode == 'train':
            df = self.dataset.train_dataset()
        else:
            df = self.dataset.test_dataset()

        coef, label = self.reshape_data(df['coef'], df['label'])

        feed = {
            self.model.x: coef,
            self.model.y: label
        }
        self.session.run(dataset_iterator.initializer, feed_dict=feed)

        return inputs, dataset_handle

    def reshape_data(self, x, y):

        coef = np.array(x.values.tolist())
        coef = np.expand_dims(coef, axis=3)
        labels = np.expand_dims(y, axis=1)

        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoded_label = onehot_encoder.fit_transform(labels)

        return coef, onehot_encoded_label



