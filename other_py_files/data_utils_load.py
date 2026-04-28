import os
import re
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler



def filter_valid_files(file_list, pattern):
    return [file for file in file_list if re.match(pattern, file)]


def preprocess_and_save_data(loaded_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loaded_files = filter_valid_files(os.listdir(loaded_dir), r'^ID\d+_\d+\.csv$')

    processed_loaded = {}
    all_loaded_gait_lengths = []
    for file in loaded_files:
        df = pd.read_csv(os.path.join(loaded_dir, file))
        for gait_num in df['Gait'].unique():
            all_loaded_gait_lengths.append(len(df[df['Gait'] == gait_num]))

    global_avg_loaded_gait_length = int(np.mean(all_loaded_gait_lengths))
    print(f"Global Average Loaded Gait Length: {global_avg_loaded_gait_length}")

    for file in loaded_files:
        df = pd.read_csv(os.path.join(loaded_dir, file))
        numbers = re.findall(r'\d+', file)

        pid, label = numbers[:2]
        label = float(label)

        task_key = f"task_{label}"
        if pid not in processed_loaded:
            processed_loaded[pid] = {}
        
        gaits_data = {}
        for gait_num in df['Gait'].unique():
            gait_df = df[df['Gait'] == gait_num].drop(columns=['Gait'])
            gait_tensor = torch.tensor(gait_df.values, dtype=torch.float32)
            gaits_data[gait_num] = gait_tensor

        processed_loaded[pid][task_key] = {
            'gaits': gaits_data, 
            'label': label
        }

    torch.save(dict(processed_loaded), os.path.join(save_dir, 'processed_loaded.pt'))
    print(f"Preprocessed data saved to {save_dir}")
    
    
    
    
    
def custom_collate_fn(batch):
    if len(batch[0]) == 3:
        x_load_list, label_list, pid_list = zip(*batch)
        x_load_batch = [torch.stack(gait_list) for gait_list in x_load_list]
        y_batch = torch.stack(label_list)
        return x_load_batch, y_batch, list(pid_list)
    else:
        x_load_list, label_list = zip(*batch)
        x_load_batch = [torch.stack(gait_list) for gait_list in x_load_list]
        y_batch = torch.stack(label_list)
        return x_load_batch, y_batch




def standardize_data(train_tasks, val_tasks, test_tasks):

    train_gait_data = [gait_tensor for task_dict in train_tasks.values() for task in task_dict.values() for gait_tensor in task['gaits'].values()]

    # Concatenate all training gait data for fitting scaler
    combined_train = torch.cat(train_gait_data, dim=0)
    scaler = StandardScaler()
    scaler.fit(combined_train.numpy())

    # Define helper to transform a task dict in-place
    def apply_scaler(tasks):
        for pid, task_dict in tasks.items():
            for task_key, task in task_dict.items():
                for gait_num, gait_tensor in task['gaits'].items():
                    task['gaits'][gait_num] = torch.tensor(
                        scaler.transform(gait_tensor.numpy()), dtype=torch.float32
                    )
                    
    apply_scaler(train_tasks)
    apply_scaler(val_tasks)
    apply_scaler(test_tasks)

    return train_tasks, val_tasks, test_tasks



class PairedTaskDataset(Dataset):
    def __init__(self, loaded_tasks):
        self.paired_tasks = []
        for pid, tasks in loaded_tasks.items():
            for task_key, task_data in tasks.items():
                x_load_gaits = list(task_data['gaits'].values())
                paired_task = {
                    'x_load': x_load_gaits,
                    'label': task_data['label'],
                    'pid': pid
                }
                self.paired_tasks.append(paired_task)

    def __len__(self):
        return len(self.paired_tasks)

    def __getitem__(self, idx):
        task = self.paired_tasks[idx]
        x_load = task['x_load']
        y = torch.tensor(int(task['label']), dtype=torch.float32)
        return x_load, y, task['pid']




def cv_data_create(val_participants, test_participants, save_dir, batch_size):
    processed_loaded = torch.load(os.path.join(save_dir, 'processed_loaded.pt'), weights_only=False)

    val_test_pids = val_participants + test_participants
    train_loaded = {pid: tasks for pid, tasks in processed_loaded.items() if pid not in val_test_pids}

    val_loaded = {pid: tasks for pid, tasks in processed_loaded.items() if pid in val_participants}
    test_loaded = {pid: tasks for pid, tasks in processed_loaded.items() if pid in test_participants}

    train_loaded, val_loaded, test_loaded = standardize_data(train_loaded, val_loaded, test_loaded)

    train_dataset = PairedTaskDataset(train_loaded)
    val_dataset = PairedTaskDataset(val_loaded)
    test_dataset = PairedTaskDataset(test_loaded)
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

