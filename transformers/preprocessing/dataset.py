from torch.utils.data import Dataset

# Create custom dataset for dataloader
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"iner" : self.data[idx], 
                "label" : self.labels[idx]}