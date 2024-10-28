import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split


# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
dataset = pd.read_csv('data.csv', sep=';')

datset_train, dataset_test = train_test_split(dataset, shuffle=True, test_size=0.25)

# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
batch_size = 27
train_dataset = t.utils.data.DataLoader(ChallengeDataset(datset_train, 'train'), batch_size=batch_size, shuffle=True)
val_dataset = t.utils.data.DataLoader(ChallengeDataset(dataset_test, 'val'), batch_size=batch_size, shuffle=False)

print('Train and Validation Datasets loaded and split')


# create an instance of our ResNet model
model_obj = model.ResNet()
print('ResNet Model object created')

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCELoss()
print('Loss Criterion object created')

# optimizer = t.optim.SGD(model_obj.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=0.0001, nesterov=True)

optimizer = t.optim.Adam(model_obj.parameters(), lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)

print('Optimizer object created')


# go, go, go... call fit on trainer
trainer = Trainer(model_obj, criterion, optimizer, train_dataset, val_dataset, cuda=True, early_stopping_patience=20)
res = trainer.fit(epochs=300)

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')