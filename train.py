import SimpleITK
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import random
import pywt
import skimage
from PIL import Image

from model import ResNet18
from featureNet import featureNet
from vit_pytorch import ViT
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau


def evaluation(total_pred, total_gt):
    aggregate_results=dict()
    aggregate_results["plcc"] = abs(pearsonr(total_pred, total_gt)[0])
    aggregate_results["srocc"] = abs(spearmanr(total_pred, total_gt)[0])
    aggregate_results["krocc"] = abs(kendalltau(total_pred, total_gt)[0])
    aggregate_results["overall"] = abs(pearsonr(total_pred, total_gt)[0]) + abs(spearmanr(total_pred, total_gt)[0]) + abs(kendalltau(total_pred, total_gt)[0])

    return aggregate_results

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

class TensorData(Dataset):
# https://didu-story.tistory.com/85
    def __init__(self, b_path, train_list, label_list):
        self.b_path = b_path
        self.train_list = train_list
        self.label_list = label_list    
        self.len = len(train_list)

    def make_wave_data(self, image):
        image = np.array(image)
        
        #image = np.pad(image, (126,126), 'constant', constant_values=(0.0))
        
        coeffs2 = pywt.dwt2(np.squeeze(image), 'bior1.3')
        
        return coeffs2

    # x,y를 튜플형태로 바깥으로 내보내기
    def __getitem__(self, index):
        #x = SimpleITK.ReadImage(self.b_path + self.train_list[index])
        #x = SimpleITK.GetArrayFromImage(x)
        x = Image.open(self.b_path + self.train_list[index])
        features = self.make_features(x)
        
        LL, (LH, HL, HH) = self.make_wave_data(x)
        
        x = np.array(x.resize((384, 384)))
        
        x = np.expand_dims(x, axis=0)
        
        LL = np.expand_dims(LL, axis=0)
        LH = np.expand_dims(LH, axis=0)
        HL = np.expand_dims(HL, axis=0)
        HH = np.expand_dims(HH, axis=0)
        
        wave_set = np.concatenate((LL,LH,HL,HH), axis=0)
        
        
        y = self.label_list[index]
        #print(str(x.max()) + ' : ' + str(x.min()))
        y = np.asarray([y])
        
        
        return x, y, wave_set, features

    def __len__(self):
        return self.len
        
    def make_features(self, image):
        image = np.array(image)*255.0

        dist = [1,3]
        angle = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        features = []
        num_feature = len(dist)*len(angle)
        num_metric = 5

        glcm = skimage.feature.graycomatrix(image.astype(np.uint8),dist, angle)

        tmp = []
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'contrast')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'dissimilarity')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'homogeneity')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'energy')).reshape((num_feature,-1)).squeeze())
        tmp.append(np.array(skimage.feature.graycoprops(glcm,'correlation')).reshape((num_feature,-1)).squeeze())
        tmp = np.array(tmp).reshape((num_feature*num_metric,-1)).squeeze()
        
        return tmp

base_path = '/home/oem/Desktop/LDCTIQAC2023/dataset/LDCTIQAG2023_train/LDCTIQAG2023_train/image/'


with open('/home/oem/Desktop/LDCTIQAC2023/dataset/LDCTIQAG2023_train/LDCTIQAG2023_train/train.json', encoding="UTF-8") as json_file:
    data_dic = json.loads(json_file.read())


val_input = np.load('./val_input.npy')
val_label = np.load('./val_label.npy')

train_input = []
train_label = []


for sample in data_dic.keys():
    if not sample in val_input:
        train_input.append(sample)
        train_label.append(data_dic[sample])

'''
train_input = random.sample(data_dic.keys(), 800)
train_label = []

for train_sample in train_input:
    train_label.append(data_dic[train_sample])

val_input = []
val_label = []
for sample in data_dic.keys():
    if not sample in train_input:
        val_input.append(sample)
        val_label.append(data_dic[sample])

np.save('./val_input.npy', np.asarray(val_input))
np.save('./val_label.npy', np.asarray(val_label))
'''

trainSet = TensorData(base_path, train_input, train_label)
train_loader = DataLoader(trainSet, batch_size= 32, shuffle=True)

valSet = TensorData(base_path, val_input, val_label)
val_loader = DataLoader(valSet, batch_size= 32, shuffle=False)


dataiter = iter(train_loader)

best_epoch = -1
best_eval = -1

epoch = 1000
vit_model = ResNet18()
#vit_model.load_state_dict(torch.load('./model/resnet18.pth'), strict=True)

feature_model = featureNet(vit_model).to(DEVICE)

learning_rate = 0.00002
optimizer = torch.optim.Adam(vit_model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)
criterion = torch.nn.MSELoss()

with open("./TextFile.txt", "w") as file:
    file.write("start~ \n")

print('model : ', sum(p.numel() for p in feature_model.parameters() if p.requires_grad))

for e in range(epoch):
    vit_model.train()
    train_losses = []
    train_pred = None
    train_label = None

    for imgs, labels, train_waves, train_features in iter(train_loader):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        train_waves = torch.tensor(train_waves, dtype=torch.float32)
        train_waves = train_waves.to(DEVICE)
        train_features = torch.tensor(train_features, dtype=torch.float32)
        train_features = train_features.to(DEVICE)

        model_output = feature_model(imgs, train_waves, train_features)
        #model_output = hardtanh(model_output)
        model_output = model_output.double()

        optimizer.zero_grad()
        loss = criterion(model_output, labels)
        train_losses.append(loss)
        loss.backward()
        optimizer.step()

        if train_pred is None:
            train_pred = model_output.detach().cpu().numpy()
            train_label = labels.detach().cpu().numpy()
        else:
            train_pred = np.concatenate((train_pred, model_output.detach().cpu().numpy()), axis=0)
            train_label = np.concatenate((train_label, labels.detach().cpu().numpy()), axis=0)

    train_pred = np.squeeze(train_pred)
    train_label = np.squeeze(train_label)
    evalu_result = evaluation(train_pred, train_label)
    with open("./TextFile.txt", "a") as file:
        file.write(str(e) + " epoch train loss: " + str(sum(train_losses)/len(train_losses)) + "\n")
        file.write(str(e) + " epoch train evaluation: " + str(evalu_result["overall"]) + "\n")

    if e % 10 == 0:
        scheduler.step()

    # validation
    val_dataiter = iter(val_loader)
    vit_model.eval()
    val_losses = []
    val_pred = None
    val_label = None

    with torch.no_grad():
        
        for v_imgs, v_labels, v_waves, v_features in iter(val_dataiter):
            
            v_imgs = v_imgs.to(DEVICE)
            v_labels = v_labels.to(DEVICE)
            v_waves = torch.tensor(v_waves, dtype=torch.float32)
            v_waves = v_waves.to(DEVICE)
            v_features = torch.tensor(v_features, dtype=torch.float32)
            v_features = v_features.to(DEVICE)

            val_output = feature_model(v_imgs,v_waves,v_features)
            #val_output = hardtanh(val_output)
            val_output = val_output.double()
            v_loss = criterion(val_output, v_labels)
            val_losses.append(v_loss)

            if val_pred is None:
                val_pred = val_output.detach().cpu().numpy()
                val_label = v_labels.detach().cpu().numpy()
            else:
                val_pred = np.concatenate((val_pred, val_output.detach().cpu().numpy()), axis=0)
                val_label = np.concatenate((val_label, v_labels.detach().cpu().numpy()), axis=0)
    
    val_pred = np.squeeze(val_pred)
    val_label = np.squeeze(val_label)
    v_evalu_result = evaluation(val_pred, val_label)
    with open("./TextFile.txt", "a") as file:
        file.write(str(e) + " epoch val loss: " + str(sum(val_losses)/len(val_losses)) + "\n")
        file.write(str(e) + " epoch val evaluation: " + str(v_evalu_result["overall"]) + "\n")

    torch.save(feature_model.state_dict(), "./featureNet_model/"+ str(e) +"_resnet18.pth")
