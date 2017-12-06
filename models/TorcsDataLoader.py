import torch
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(path_to_project_root, 'data/10-laps')

class TorcsTrackDataset(Dataset):
    """Represents data from racing a particular track in TORCS"""

    def __init__(self, csv_files):
        data = []
        print("Initializing using files:")
        for file_name in csv_files:
            print(file_name)
            track = pd.read_csv(os.path.join(data_dir, file_name), sep=";", index_col=False, header=0)

            #FOR WINNER DATA:
            #track = pd.read_csv(os.path.join(data_dir, file_name), index_col=False, header=0)
            data.append(track)

        data = pd.concat(data)

        ####### For our generated data:
        speed_x  = data.iloc[:, 11].values
        t_center = data.iloc[:, 14].values
        t_angle  = data.iloc[:, 3].values
        t_edges  = data.iloc[:, 20:39].values


        # Only 3 outer edges, left and right {-90, -75, -60, +60, +75, +90}
        #t_edges_neg = data.iloc[:, 20:28].values
        #t_edges_pos = data.iloc[:, 31:39].values
        #t_edges = np.append(t_edges_neg, t_edges_pos, axis=1)

        # Improved frontal sensors
        #t_edges_improved = np.amax(data.iloc[:, 29:32].values, axis=1)

        ####### For data from winner bot 2016:
        #speed_x  = data.iloc[:, 3].values
        #t_center = data.iloc[:, 4].values
        #t_angle  = data.iloc[:, 5].values
        #t_edges  = data.iloc[:, 6:].values

        #states = np.stack((speed_x, t_edges_improved), axis=-1)
        states = np.stack((speed_x, t_center, t_angle), axis=-1)

        self.carstates = np.append(states, t_edges, axis=1)
        self.x_dim = self.carstates.shape[1]

        #accel = data.iloc[:, 0].values
        brake = data.iloc[:, 1].values
        #accel_brake = accel - brake

        #steer = data.iloc[:, 2].values

        self.targets = brake #np.stack((accel, brake), axis=-1)

        self.t_dim = 1 #self.targets.shape[1]

    def __len__(self):
        return len(self.track_frame)

    def __getitem__(self, idx):
        datapoint = self.carstates[idx]
        target = self.targets[idx]

        return {'input' : datapoint, 'target' : target}

class AalborgDataset(TorcsTrackDataset):
    def __init__(self):
        path = os.path.join(*PATH_AALBORG)
        super().__init__(path)

def save_parameters(model, label):
    in_dim, hidden, out_dim = model.get_n_units()

    saved_models = os.path.join(path_to_project_root, 'saved_models')
    model_name = str(type(model)).split('.')[-1]
    print(model_name)
    model_name = model_name[:-2]

    torch.save(model.state_dict(), os.path.join(saved_models,
                                    '{}_{}-{}-{}_{}.h5'.format(label, in_dim, hidden, out_dim, model_name)))
