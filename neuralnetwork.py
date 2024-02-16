import os
import numpy as np
import sys
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tdata
import torch.backends.mps
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import re
import seaborn

def parse(line):
    line = str(line)
    regex = r"[-+]?\d*\.\d+|\d" # searches for all floats or integers
    list = re.findall(regex, line)
    output = [float(x) for x in list]
    return output

## NEURAL NETWORK CLASSES
class FC_FF_NN_3LAYER(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size) -> None:
        super(FC_FF_NN_3LAYER, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size) # Layer 1
        self.relu1 = nn.ReLU() # Activation 1
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # Layer 2
        self.relu2 = nn.ReLU() # Activation 2
        self.fc3 = nn.Linear(hidden2_size, hidden3_size) # Layer 3
        self.relu3 = nn.ReLU() # Activation 3
        self.fc4 = nn.Linear(hidden3_size, output_size) # Layer 4
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        return out

class FC_FF_NN_4LAYER(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, hidden4_size, output_size) -> None:
        super(FC_FF_NN_4LAYER, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size) # Layer 1
        self.relu1 = nn.ReLU() # Activation 1
        self.fc2 = nn.Linear(hidden1_size, hidden2_size) # Layer 2
        self.relu2 = nn.ReLU() # Activation 2
        self.fc3 = nn.Linear(hidden2_size, hidden3_size) # Layer 3
        self.relu3 = nn.ReLU() # Activation 3
        self.fc4 = nn.Linear(hidden3_size, hidden4_size) # Layer 4
        self.relu4 = nn.ReLU() # Activation 4
        self.fc5 = nn.Linear(hidden4_size, output_size) # Layer 5

    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out

## BUILD DATASET OF TENSORS
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.targets = torch.from_numpy(targets).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index,:], self.targets[index,:]
## Ask to use GPU or MPS if possible
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

def build_model(len_pattern=16,firstlayer=32,secondlayer=32,thirdlayer=16,fourthlayer=16,coords=2, model_layers=4):
    if model_layers==3:
        model = FC_FF_NN_3LAYER(len_pattern,firstlayer,secondlayer,thirdlayer,coords).to(device)
    elif model_layers==4:
        model = FC_FF_NN_4LAYER(len_pattern,firstlayer,secondlayer,thirdlayer,fourthlayer,coords).to(device)
    return model

def load_model(model_dir):
    model = build_model()
    if model_dir.endswith(".pt"):
        model_path=model_dir
        model.load_state_dict(torch.load(model_path))
    else:
        dir_list = os.listdir(model_dir)
        model_path = ""
        no_model=0
        for item in dir_list:
            if(item.endswith(".pt")):
                print(item)
                no_model+=1
        if(no_model==0):
            print("No Models in Folder. Exiting.")
            sys.exit()
        else:
            model_path = input("Select model: \n")
        if model_dir.endswith(".pt"):
            model.load_state_dict(torch.load(model_path))
    return model

def EuclideanDistance(a, b):
    d = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return d

def train_model(model, train_DL,test_DL, epochs, criterion, optimizer):
    accuracy_test=[]
    accuracy_train=[]
    
    for ep in range(epochs):
        model.train()
        n_samples = len(train_DL)
        train_loss = 0.0
        #n_samples = 0.0
        for patterns, coords in train_DL:
            # send patterns and coordinates to device
            patterns, coords = patterns.to(device), coords.to(device)

            # forward pass
            outputs = model(patterns.float())
            loss = criterion(outputs, coords.float())

            # backwards pass and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate loss
            train_loss += loss.item() * patterns.size(0)
            #n_samples += patterns.size(0)
        
        # calculate average training loss
        train_loss /= n_samples
        if ((ep+1) % 50 == 0):
            print(f"Epoch {ep+1}/{epochs}, Train Loss: {train_loss:.4f}")
        accuracy_train.append((1.0-train_loss)*100)
        
        model.eval()
        n_samples = len(test_DL)
        test_loss = 0.0
        for patterns, coords in test_DL:
            # send patterns and coordinates to device
            patterns, coords = patterns.to(device), coords.to(device)

            # forward pass
            with torch.no_grad():
                outputs = model(patterns.float())
                loss = criterion(outputs, coords.float())

            # accumulate loss
            test_loss += loss.item() * patterns.size(0)

        test_loss /= n_samples
        if ((ep+1) % 50 == 0):
            print(f"Epoch {ep+1}/{epochs}, Test Loss: {test_loss:.4f}")
        accuracy_test.append((1.0-test_loss)*100)
        
    return model, np.average(accuracy_train), np.average(accuracy_test)

def train_loop(model, train_DL, criterion, optimizer):
    size = len(train_DL.dataset)
    for batch, (patterns, coords) in enumerate(train_DL):
        # Compute prediction and loss
        pred=model(patterns)
        loss=criterion(pred,coords)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        """ if batch % 25 == 0:
            _loss, current = loss.item(), (batch+1)*len(patterns)
            print(f"loss: {_loss:.4f} [{current:>5d}/{size:>5d}]") """

def test_loop(model, test_DL, criterion):
    size = len(test_DL.dataset)
    num_batches = len(test_DL)
    test_loss = 0.0
    correct = []

    with torch.no_grad():
        for patterns, coords in test_DL:
            pred=model(patterns)
            test_loss += criterion(pred,coords).item()
            predicted = pred.detach().numpy()
            c = coords.detach().numpy()
            for i in range(len(c)):
                correct.append(EuclideanDistance(predicted[i], c[i]))
    
    test_loss /= num_batches
    correct_avg = np.average(correct)
    
    #print(f"Test Error: EuclidDist:{correct_avg:.4f}, Avg loss: {test_loss:.4f}")

def plot_points_with_lines(data1, data2):
    """ plt.scatter([point[0] for point in data1], [point[1] for point in data1], marker='.', color='blue')
    plt.scatter([point[0] for point in data2], [point[1] for point in data2], marker='x', alpha=0.8, color='green') """
    plt.figure(figsize=(9,9))
    x=[]
    num_points = min(len(data1), len(data2))
    for i in range(num_points):
        distance = EuclideanDistance(data1[i], data2[i])
        x.append(distance)
        if distance < 0.1:    
            plt.plot([np.array(data1[i])[0], np.array(data2[i])[0]],
                     [np.array(data1[i])[1], np.array(data2[i])[1]],
                     color='dimgrey', linewidth=0.4, alpha=0.7)
            plt.scatter(data1[i][0], data1[i][1], marker='2', color='red')
            plt.scatter(data2[i][0], data2[i][1], marker='.', color='black')
        """ else: 
                plt.scatter(data1[i][0], data1[i][1], marker='x', color='grey', alpha=0.6)
                plt.scatter(data2[i][0], data2[i][1], marker='.', color='grey', alpha=0.6) """
            
    #plt.hist(x, bins=20)
    plt.gca().set_aspect('equal')
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.setp(plt.gca().get_yticklabels(), visible=False)
    plt.title('OnsDen_fW Predictions in Rhythm Space')
    plt.legend(["d<0.1",'OnsDen_fW', 'Rhythm Space'], loc='upper right', fontsize='small')

    plt.show()

def mean_euclidean_distance(y_true, y_pred):
    distances = [EuclideanDistance(true, pred) for true, pred in zip(y_true, y_pred)]
    return np.mean(distances)

def NN_pipeline(patterns, coords, _save, model_dir, _load=False):

    # Load or Build Model
    if _load:
        model = load_model(model_dir)
    else:
        model = build_model()
    
    seed = np.random.randint(0,1000)

    if _load==False:
        # Train
        learning_rate = 0.001
        batch_size = 32
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epochs = 200
        train_loss = 0.0
        test_loss = 0.0

        # train test split
        patterns = np.asarray(patterns, dtype=np.float32)
        coords = np.asarray(coords, dtype=np.float32)
        train, test, train_coords, test_coords = train_test_split(patterns, coords, test_size=0.2, random_state=seed)
        train_dataset = CustomDataset(train, train_coords)
        test_dataset = CustomDataset(test, test_coords)
        train_DL = tdata.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_DL = tdata.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print_int=50
        for ep in range(epochs):
            if ep%print_int==0 or ep==epochs-1:
                #print(f"EPOCH: {ep}/{epochs}")
                test_loop(model, test_DL, criterion)
            train_loop(model, train_DL, criterion, optimizer)
        
    predicted = []
    vals=[0.0 for x in range(3)]
    distance = []
    threshold_bins = [0 for x in range(8)]
    _threshold_bins = [0 for x in range(8)]

    for i in range(len(patterns)):
        row = torch.Tensor(patterns[i]).float()
        pred = model(row)
        predicted.append(pred.detach().numpy())
        distance.append(EuclideanDistance(predicted[i],coords[i]))


        if distance[i]<=0.025:
            threshold_bins[0] += 1
            _threshold_bins[0] += 1
        if distance[i]<=0.05:
            threshold_bins[1] += 1
            if distance[i]>0.025:
                _threshold_bins[1] += 1
        if distance[i]<=0.075:
            threshold_bins[2] += 1
            if distance[i]>0.05:
                _threshold_bins[2] += 1
        if distance[i]<=0.1:
            threshold_bins[3] += 1
            if distance[i]>0.075:
                _threshold_bins[3] += 1
        if distance[i]<=0.15:
            threshold_bins[4] += 1
            if distance[i]>0.1:
                _threshold_bins[4] += 1
        if distance[i]<=0.2:
            threshold_bins[5] += 1
            if distance[i]>0.15:
                _threshold_bins[5] += 1
        if distance[i]<=0.25:
            threshold_bins[6] += 1
            if distance[i]>0.20:
                _threshold_bins[6] += 1
        if distance[i]>0.25:
            _threshold_bins[7] += 1    
        threshold_bins[7] += 1

        #if i!=0 and i%200==0:
            #print(f"Pred:[{predicted[i][0]:.3f},{predicted[i][1]:.3f}]-->Actual:[{coords[i][0]:.3f},{coords[i][1]:.3f}] <> Dist:{distance[i]:.3f}")
    print(f"\n|{threshold_bins[0]/len(patterns):.3f}|{threshold_bins[1]/len(patterns):.3f}|{threshold_bins[2]/len(patterns):.3f}|{threshold_bins[3]/len(patterns):.3f}|{threshold_bins[4]/len(patterns):.3f}|{threshold_bins[5]/len(patterns):.3f}|{threshold_bins[6]/len(patterns):.3f}|{threshold_bins[7]/len(patterns):.3f}| <-- CUMULATIVE")
    print(f"|{_threshold_bins[0]/len(patterns):.3f}|{_threshold_bins[1]/len(patterns):.3f}|{_threshold_bins[2]/len(patterns):.3f}|{_threshold_bins[3]/len(patterns):.3f}|{_threshold_bins[4]/len(patterns):.3f}|{_threshold_bins[5]/len(patterns):.3f}|{_threshold_bins[6]/len(patterns):.3f}|{_threshold_bins[7]/len(patterns):.3f}| <-- SEPARATE")
    print(f"|-----|-----|-----|-----|-----|-----|-----|-----|")
    print(f"|0.025|0.050|0.075|0.100|0.150|0.200|0.250|1.000| <-- DISTANCE BINS") 
    #print(f"\n >>>>>>>>>>>>>>>>>>>>>>> Euclidean Distance = {np.mean(distance):.5f}[{np.std(distance):.3f}]var:{np.var(distance):.3f} <<<<<<<<< CV{np.std(distance) / np.mean(distance):.5f}")
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = []
    predicted_coords=np.array(predicted)
    target_coords=np.array(coords)
    for train_index, test_index in kf.split(predicted_coords):
        train_preds, test_preds = predicted_coords[train_index], predicted_coords[test_index]
        train_targets, test_targets = target_coords[train_index], target_coords[test_index]

        score = mean_euclidean_distance(test_targets, test_preds)
        scores.append(score)
    """ # cross eval
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(None, patterns, coords, scoring=make_scorer(mean_euclidean_distance), cv=cv, n_jobs=-1)"""
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


    if _save:
        torch.save(model.state_dict(), model_dir+".pt")
    CV = np.std(distance) / np.mean(distance)
    vals = [round(np.mean(distance),4), round(np.std(distance), 4),round(CV, 4)]
    #plot_points_with_lines(predicted, coords)
    return predicted, vals


""" K-fold cross eval stuff
    #model, acc_train, acc_test = train_model(model, train_DL,test_DL, epochs, criterion, optimizer)
    #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    #scores = cross_val_score(model, patterns, coords, scoring='accuracy', cv=cv, n_jobs=-1)
    # report result
    #print('Mean Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
 """