import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt('saved/training_data.csv', delimiter=',')
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data

        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(self.data) #fits and transforms
        pickle.dump(self.scaler, open("saved/scaler.pkl", "wb")) #save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        #pass
        return len(self.data)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        # STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
        # x and y should both be of type float32. There are many other ways to do this, but to work with autograding
        # please do not deviate from these specifications.
        x = self.normalized_data[idx, :-1].astype('float32')
        y = self.normalized_data[idx, 6].astype('float32')
        return {'input':x, 'label':y}
        

class Data_Loaders():
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        # STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
        # make sure your split can handle an arbitrary number of samples in the dataset as this may vary
        #print(self.nav_dataset.normalized_data[:,6])
        train_indices, test_indices, _, _ = train_test_split(
        range(len(self.nav_dataset.normalized_data)),
        self.nav_dataset.normalized_data[:,6],
        stratify=self.nav_dataset.normalized_data[:,6],
        test_size=0.2,
        random_state=42
        )

        #train_split = Subset(self.nav_dataset.normalized_data, train_indices)
        train_split = [self.nav_dataset.__getitem__(train_indices[0])]
        for i in range(1, len(train_indices)):
            train_split.append(self.nav_dataset.__getitem__(train_indices[i]))

        #test_split = Subset(self.nav_dataset.normalized_data, test_indices)
        test_split = [self.nav_dataset.__getitem__(test_indices[0])]
        for i in range(1, len(test_indices)):
            test_split.append(self.nav_dataset.__getitem__(test_indices[i]))
            
        self.train_loader = DataLoader(train_split, batch_size= batch_size)
        self.test_loader = DataLoader(test_split, batch_size= batch_size)
        #i = 0
        #for idx in range(len(train_split)):
        #    if train_split[idx]['label'] == 1:
        #        i = i+1
        #        print(i)
            

def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample['input'], sample['label']
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample['input'], sample['label']

if __name__ == '__main__':
    main()
