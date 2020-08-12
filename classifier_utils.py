import torch
import vision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class classifier_data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, male_neutral, female_neutral, male_smiling, female_smiling, transform=None, val=False, cut_len=0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.val = val
        if self.val:
            cut_len = len(male_smiling)
        self.idx = range(1, cut_len + 1)
        
        self.idx_all = []

        for i in self.idx:
            self.idx_all.append((i, 0))
            self.idx_all.append((i, 1))
            self.idx_all.append((i, 2))
            self.idx_all.append((i, 3))

        if self.val:
          self.male_neutral, self.female_neutral, self.male_smiling, self.female_smiling = male_neutral[-cut_len:], female_neutral[-cut_len:], male_smiling[-cut_len:], female_smiling[-cut_len:]
        else:  
          self.male_neutral, self.female_neutral, self.male_smiling, self.female_smiling = male_neutral[:cut_len], female_neutral[:cut_len], male_smiling[:cut_len], female_smiling[:cut_len]

    def __len__(self):
        return len(self.idx_all)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # print("idx: "+str(idx)+"\n.\n")

        n, r = self.idx_all[idx]

        all_here = self.male_neutral[:n], self.female_neutral[:n], self.male_smiling[:n], self.female_smiling[:n]

        imgs_all = torch.zeros(int(len(self.idx_all) / 4), 3, 64, 64)
        for i, each in enumerate(all_here[r]):

            img = Image.open(each).convert("RGB")

            if self.transform:
                img = self.transform(img)

            imgs_all[i,:,:,:] = img

        return imgs_all, n, r


class enc_classifier(nn.Module):
   def __init__(self):
      super().__init__()
      self.conv = nn.Conv2d(3, 16, 3)
      # self.resnet = resnet50(num_classes=768)
      self.fc_0 = nn.Linear(61504, 2048)
      self.fc_1 = nn.Linear(2048, 2048)
      self.fc_2 = nn.Linear(2048, 4)


   def forward(self, data, n, feat_in = None):
      data = data.squeeze(0)[:n,:,:,:]
      x = self.conv(data)
      x = F.dropout(x)
      x = x.flatten(start_dim = 1)
      x = F.relu(x)
      x = self.fc_0(x)
      x = F.dropout(x)
      x = F.relu(x)
      x = x.max(dim = 0)[0]
      x = F.relu(x)
      
      feat_org = self.fc_1(x)
      feat = F.relu(feat_org)
      
      final = self.fc_2(feat)
      final = F.softmax(final)
      
      return feat.reshape(1, -1), final.reshape(1, -1)
