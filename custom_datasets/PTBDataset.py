from torch.utils.data import Dataset
import json
import pandas as pd
import wfdb
import numpy as np
from typing import Dict, Optional
import os
import ast
from sklearn.preprocessing import MinMaxScaler

class PTBDataset(Dataset):
    def __init__(self, partition_path: str, ds_path: str, is_train: bool, pred_len: int, context_len: int, partition_id: Optional[int] = None, filter_site: Optional[int] = None):
        self.is_train = is_train
        self.ds_path = ds_path
        self.pred_len = pred_len
        self.context_len = context_len

        self.partition_info = self.get_partition_info(partition_path)

        self.filter_site = filter_site

        self.ptb_df = self.get_ptb_df()

        self.partition_id = partition_id

        if self.is_train and not self.partition_id:
            raise Exception("parititon_id is required in train mode")
    
    def get_partition_info(self, partition_path: str) -> Dict:        
        with open(partition_path, "r") as f:
            partition_info = json.load(f)
        
        return partition_info
    
    def get_ptb_df(self) -> pd.DataFrame:
        patient_df = pd.read_csv(os.path.join(self.ds_path, "ptbxl_database.csv"))
        patient_df.scp_codes = patient_df.scp_codes.apply(lambda x: ast.literal_eval(x))


        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in mapping_df.index and y_dic[key] == 100:
                    tmp.append(mapping_df.loc[key].diagnostic_class)
            return list(set(tmp))
        
        # Read mapping
        mapping_df = pd.read_csv(
            os.path.join(self.ds_path, "scp_statements.csv"), 
            index_col=0
        )
        mapping_df = mapping_df[mapping_df.diagnostic == 1]

        # Apply diagnostic superclass
        patient_df['diagnostic_superclass'] = patient_df.scp_codes.apply(aggregate_diagnostic)

        if self.is_train:
            return patient_df.iloc[self.partition_info["train_idx"]].reset_index()
        elif not self.filter_site: 
            return patient_df.iloc[self.partition_info["test_idx"]].reset_index()
        else:
            patient_df = patient_df.iloc[self.partition_info["test_idx"]]
            return patient_df[patient_df["site"] == self.filter_site].reset_index()

    def __len__(self):
        if self.is_train:
            return self.partition_info["partition"][self.partition_id]["partition_size"]
        else:
            return len(self.ptb_df) * (5000 - self.context_len - self.pred_len + 1)
        
    def get_ecg_signal(self, fpath: str):
        record = wfdb.rdrecord(fpath)
        channel_index = record.sig_name.index('II')
        return record.p_signal[:, channel_index]


    def getitem_train(self, index):
        random_start = np.random.choice(np.arange(5000 - self.context_len - self.pred_len + 1))
        record_index = self.partition_info["partition"][self.partition_id]["partition_idx"][index]

        record_fpath = os.path.join(self.ds_path, self.ptb_df["filename_hr"][record_index])

        ecg_signal = self.get_ecg_signal(record_fpath)
        
        scaler = MinMaxScaler()
        model = scaler.fit(ecg_signal.reshape((-1, 1)))
        ecg_signal = model.transform(ecg_signal.reshape((-1, 1))).reshape((-1))

        item = ecg_signal[random_start: (random_start + self.context_len + self.pred_len)]

        return item[:self.context_len], item[self.context_len:]
    
    def getitem_test(self, index):
        record_index = index // (5000 - self.context_len - self.pred_len + 1)
        signal_start_index = index % (5000 - self.context_len - self.pred_len + 1)

        record_fpath = os.path.join(self.ds_path, self.ptb_df["filename_hr"][record_index])

        ecg_signal = self.get_ecg_signal(record_fpath)

        scaler = MinMaxScaler()
        model = scaler.fit(ecg_signal.reshape((-1, 1)))
        ecg_signal = model.transform(ecg_signal.reshape((-1, 1))).reshape((-1))

        item = ecg_signal[signal_start_index: (signal_start_index + self.context_len + self.pred_len)]

        return item[:self.context_len], item[self.context_len:]

    def __getitem__(self, index):
        if self.is_train:
            return self.getitem_train(index)
        else:
            return self.getitem_test(index)