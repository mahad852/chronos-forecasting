from torch.utils.data import Dataset

from typing import List

import os

import scipy.io

from sklearn.preprocessing import StandardScaler

class VitalSignsDataset(Dataset):
    def __init__(self, 
                 user_ids : List[str], 
                 data_attribute : str, 
                 scenarios : List[str],
                 context_len = 600, 
                 pred_len = 60, 
                 data_path = '/home/mali2/datasets/vital_signs',
                 is_train = True):

        self.root_folder = data_path        
        self.validate_path(self.root_folder, f"Incorrect data_path supplied. Expected a directory, {self.root_folder} is not a directory.")
        
        self.user_data_paths = self.get_user_data_paths(user_ids, scenarios)

        self.context_len = context_len
        self.pred_len = pred_len

        self.data_attribute = data_attribute

        self.is_train = is_train
        self.total_len = 0

        self.data_file = None
        self.data = None

        self.set_indices()


    def __len__(self):
        return self.total_len

    def get_user_data_paths(self, user_ids: List[str], scenarios: List[str]) -> List[str]:
        user_data_paths = []
        for user_id in user_ids:
            for scenario in scenarios:
                user_data_paths.append(self.get_user_data_path(user_id, scenario))

        return user_data_paths
    
    def get_user_data_path(self, user_id :str, scenario: str):        
        user_folder_path = os.path.join(self.root_folder, user_id)
        self.validate_path(user_folder_path, f"Expected {user_folder_path} to be a path to a directory. Are the data_path and user_ids correct?")
        
        for file in os.listdir(user_folder_path):
            if ".mat" in file and scenario.lower() in file.lower():
                return os.path.join(user_folder_path, file)
                
        raise ValueError(f"Expected {user_id} folder to have a \".mat\" file for the scenario \"{scenario}\", but found no such file in {user_folder_path}.")
    
    def validate_path(self, path: str, error_msg: str):
        if not os.path.exists(path):
            raise ValueError(error_msg)

    def get_data(self, file):
        if self.data_file != file:
            self.clear_data()

            self.data_file = file
            self.data = scipy.io.loadmat(file)[self.data_attribute].reshape(-1)

            total = self.data.shape[0]
            if self.is_train:
                self.data = self.data[:int(total * 0.60)]
            else:
                self.data = self.data[int(total * 0.60):]
            
            scaler = StandardScaler()
            model = scaler.fit(self.data)
            self.data = model.transform(self.data)

        return self.data
    
    def clear_data(self):
        del self.data_file
        self.data_file = None
        
        del self.data
        self.data = None
        
    def __getitem__(self, index):
        file = self.index_to_file[index]
        file_index = index - self.file_to_start_index[file]
        
        data = self.get_data(file)
        
        return data[file_index : file_index + self.context_len], data[file_index + self.context_len : file_index + self.context_len + self.pred_len]

    def set_indices(self):
        self.index_to_file = {}
        self.file_to_start_index = {}

        self.total_len = 0
        for file in self.user_data_paths:
            data = self.get_data(file)

            prev = self.total_len
            self.total_len += (data.shape[0] - (self.context_len + self.pred_len) + 1)
            
            self.file_to_start_index[file] = prev
            
            for index in range(prev, self.total_len):
                self.index_to_file[index] = file

        self.clear_data()