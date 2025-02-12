import torch
import numpy as np
import os




class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, sample_dir, embed_dim = 384, train = False, time = 2):
        """
        Class to create embedding dataset from extracted embeddings
        train = False --> Don't duplicate training set
        time = number of time to dupicate
        """
        self.sample_dir = sample_dir
        self.n_segments = len(os.listdir(self.sample_dir))
        self.data = np.zeros((self.n_segments, embed_dim))
        
        # Sorted segment based on start_time
        self.sorted_segments = sorted(os.listdir(sample_dir), key= lambda x: float(x.split("_")[-2]))

        # Assign segments
        for idx, segment_npy in enumerate(self.sorted_segments):
            segment_path = self.sample_dir + "/" + segment_npy
            segment_embed = np.load(segment_path)
            self.data[idx] = segment_embed
            
        if train:
            for time in range(time):
                self.data = np.concatenate((self.data, self.data), axis = 0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()
        return sample
    

class AutoEncoderDataset(torch.utils.data.Dataset):
    """
    Create dataset from predefined tensor for each autoencoder in MOE
    """
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
    
    


    
