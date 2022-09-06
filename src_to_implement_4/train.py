import torch as t
from torch.utils.data import DataLoader
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
file = pd.read_csv('data.csv', sep=';')
file = file.to_numpy()
train, val = train_test_split(file, test_size=0.1)
print(len(train))
print(len(train[0]))
train = pd.DataFrame(data=train, columns=['filename', 'crack', 'inactive'])
val = pd.DataFrame(data=val, columns=['filename', 'crack', 'inactive'])



# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
train_data = ChallengeDataset(train, 'train')
train_dataloader = DataLoader(train_data,
                              batch_size=100,
                              shuffle=True
                              )
val_data = ChallengeDataset(val, 'val')
val_dataloader = DataLoader(val_data,
                            batch_size=32,
                            shuffle=True
                            )
#print(val_dataloader[0])


# create an instance of our ResNet model and try some models
model = model.ResNet(pool='avg')
#model = model.ResNet2()
#model = model.Pre_resnet()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.MSELoss()

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
model = model.to(device)
params = model.parameters()
# for param in params:
#     param.requires_grad = False
optimizer = t.optim.Adam(params=params, lr=8e-05, weight_decay=0.00001)
#optimizer = t.optim.SGD(params=params, lr=0.0001, momentum=0.9, weight_decay=0.00001)

training = Trainer(model, criterion, optimizer, train_dataloader, val_dataloader, cuda=True, early_stopping_patience=30)

# go, go, go... call fit on trainer
res = training.fit(epochs=50)

#training.restore_checkpoint(26)
#training.save_onnx(r'C:\Users\Summer\Desktop\src_to_implement\checkpoint_{:03d}.onnx'.format(26))

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')