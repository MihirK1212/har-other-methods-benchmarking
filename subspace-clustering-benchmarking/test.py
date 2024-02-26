import warnings
from sklearn.metrics import accuracy_score

import data.read as data_read
import data.utils as data_utils
import constants
from ssc import SSC


if __name__ == "__main__":

    data, labels, subjects = data_read.read_msr_data(data_dir=constants.MSR_ACTION_3D_DATA_DIR)
    data, labels, subjects = data_utils.remove_anomalies(data=data, labels=labels, subjects=subjects)

    lb, ub = 40, 60
    data, labels, subjects = data[lb:ub], labels[lb:ub], subjects[lb:ub]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dam = SSC()
        dam.load_training_config()
        predictions = dam.get_predictions(data, labels)
        print('Predictions:', predictions)
        print('Labels:', labels)
        print('Accuracy:', accuracy_score(predictions, labels))
