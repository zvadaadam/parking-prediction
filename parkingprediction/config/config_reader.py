import os
import yaml

## define custom tag handler
def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


class ConfigReader(object):

    def __init__(self, config_path='/Users/adamzvada/Documents/hackathon/iotea/ParkingPrediction/config/test.yml'):

        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        yaml.add_constructor('!join', join)

        with open(config_path, 'r') as f:
            config = yaml.load(f)

            self.info = config['info']
            self.dataset = config['dataset']
            self.hyperparams = config['hyperparams']
            self.wavelet = config['wavelet']
            self.model = config['model']

    def model_name(self):
        return self.info['model_name']

    # --DATASET--

    def csv_path(self):
        path = self.dataset['csv_path']

        return self._absolute_path(path)

    def num_samples(self):
        return self.dataset['num_samples']

    def sample_rate(self):
        return self.dataset['sample_rate']

    def max_period(self):
        return self.dataset['max_period']

    def stochastic_ratio(self):
        return self.dataset['stochastic_ratio']

    def signal_length(self):
        return self.dataset['signal_length']

    def test_size(self):
        return self.dataset['test_size']

    def coef_preprocess(self):
        return self.dataset['coef_preprocess']

    # --HYPERPARAMETERS--

    def feature_size(self):
        return self.hyperparams['feature_size']

    def batch_size(self):
        return self.hyperparams['batch_size']

    def num_layers(self):
        return self.hyperparams['num_layers']

    def num_classes(self):
        return self.hyperparams['num_classes']

    def num_epoches(self):
        return self.hyperparams['num_epoches']

    def num_iterations(self):
        return self.hyperparams['num_iterations']


    # --WAVELET--

    def wavelet_name(self):
        return self.wavelet['name']

    def max_scale(self):
        return self.wavelet['max_scale']

    # --MODEL--

    def tensorboard_path(self):
        path = self.model['tensorboard_path']

        return self._absolute_path(path)

    def trained_model_path(self):
        path = self.model['trained_path']

        return self._absolute_path(path)

    def model_description(self):
        return self.model['model_description']

    def restore_trained_model(self):
        path = self.model['restore_trained_model']

        if path == None:
            return None

        return self._absolute_path(path)

    def _absolute_path(self, path):

        if not os.path.isabs(path):
            return os.path.join(self.ROOT_DIR, path)

        return path

if __name__ == '__main__':

    config = ConfigReader()

    print(config.model_description())

    print(config.trained_path())
