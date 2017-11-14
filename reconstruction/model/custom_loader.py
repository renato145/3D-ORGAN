import os, pickle
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from ..utils.data_prep import get_fractured

class CustomLoader(data.Dataset):
    def __init__(self, file, out_folder, subset='train', **kwargs):
        self.file = file
        self.out_folder = out_folder
        self.subset = subset
        self.data = np.load(file).item()[self.subset]
        self.load_labels()
        self.fracture_opts = kwargs
        
    def load_labels(self):
        le_file = os.path.join(self.out_folder, 'label_encoder.pkl')
        if os.path.exists(le_file):
            le = pickle.load(open(le_file, 'rb'))
            labels = le.transform(self.data['labels'])
        else:
            le = LabelEncoder()
            labels = le.fit_transform(self.data['labels'])
            pickle.dump(le, open(le_file, 'wb'))
            
        self.le = le
        self.labels = labels

    def __getitem__(self, index):
        label = self.labels[index]
        shape_target = self.data['data'][index].astype(np.float32)
        shape_source = get_fractured(shape_target, **self.fracture_opts)
        
        # from [0, 1] space to [-1, 1]
        idxs = np.argwhere(shape_source == 0)
        shape_source[idxs[:,0], idxs[:,1], idxs[:,2]] = -1
        idxs = np.argwhere(shape_target == 0)
        shape_target[idxs[:,0], idxs[:,1], idxs[:,2]] = -1
        
        return shape_source, shape_target, label

    def __len__(self):
        return len(self.data['labels'])
