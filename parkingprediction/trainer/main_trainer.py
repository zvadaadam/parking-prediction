import os, sys
import tensorflow as tf
from parkingprediction.dataset.parking_dataset import ParkingDataset
from parkingprediction.model.cnn_model import RNNModel
from parkingprediction.trainer.parking_trainer import ParkingTrainer


def main_train(config):

    dataset = SignalDataset(config)

    session = tf.Session()

    # TODO: init the right model from config
    model = CNNModel(config)

    trainer = Trainer(session, model, dataset, config)

    trainer.train()


if __name__ == '__main__':

    from parkingprediction.config.config_reader import ConfigReader

    sys.path.append(os.path.abspath('../../'))

    config_path = '/Users/adamzvada/Documents/hackathon/iotea/ParkingPrediction/config/test.yml'

    main_train(ConfigReader(config_path))
