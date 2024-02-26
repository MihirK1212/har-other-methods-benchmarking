import os
import numpy as np

import constants

def get_label_from_file_name(data_dir, file_name, sep_char):
    fnametostr = ''.join(file_name).replace(data_dir, '')
    ind = int(fnametostr.index(sep_char))
    label = int(fnametostr[ind + 1:ind + 3])
    label-=1
    return label

def read_msr_data(data_dir):
    data, labels, subjects = [], [], []
    filenames = np.array([os.path.join(data_dir, d) for d in sorted(os.listdir(data_dir))])
    for sequence_file in filenames:
        sequence = np.loadtxt(sequence_file, dtype=np.float64)[:, :3]
        num_frames = (sequence.shape[0])//20
        sequence = sequence.reshape((num_frames, constants.NUM_JOINTS, 3))
        label = get_label_from_file_name(data_dir=data_dir, file_name=sequence_file, sep_char='a')
        subject = get_label_from_file_name(data_dir=data_dir, file_name=sequence_file, sep_char='s')
        data.append(sequence)
        labels.append(label)
        subjects.append(subject)
    return data, labels, subjects