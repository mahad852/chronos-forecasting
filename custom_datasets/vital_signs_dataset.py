from torch.utils.data import Dataset

from typing import List

import os

import scipy.io

from sklearn.preprocessing import MinMaxScaler

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

        self.is_train = is_train

        self.data_attribute = data_attribute
        self.context_len = context_len
        self.pred_len = pred_len

        self.validate_path(self.root_folder, f"Incorrect data_path supplied. Expected a directory, {self.root_folder} is not a directory.")
        
        self.user_data_paths = self.get_user_data_paths(user_ids, scenarios)
        self.data_lens = [len(self.get_data(user_data_path)) for user_data_path in self.user_data_paths]

        self.start_indices = [0]
        for i in range(1, len(self.data_lens)):
            self.start_indices.append(self.start_indices[i - 1] + (self.data_lens[i - 1] - self.context_len - self.pred_len + 1))

    def __len__(self):
        return sum([data_len - self.pred_len - self.context_len + 1 for data_len in self.data_lens])
        
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
        data = scipy.io.loadmat(file)[self.data_attribute].reshape(-1)

        total = data.shape[0]
        if self.is_train:
            data = data[:int(total * 0.60)]
        else:
            data = data[int(total * 0.60):]
        
        scaler = MinMaxScaler()
        model = scaler.fit(data.reshape((-1, 1)))
        data = model.transform(data.reshape((-1, 1))).reshape((-1))

        return data
    
        
    def __getitem__(self, index):
        file_index = len(self.start_indices)
        for i in range(len(self.start_indices) - 1):
            if index >= self.start_indices[i] and index < self.start_indices[i + 1]:
                file_index = i
            
        data = self.get_data(self.user_data_paths[file_index])
        start_index = index - self.start_indices[file_index]
        
        return data[start_index : start_index + self.context_len], data[start_index + self.context_len : start_index + self.context_len + self.pred_len]