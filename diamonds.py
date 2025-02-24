import kagglehub
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def fetch_data(file_directory = 'shivam2503/diamonds', file_name = 'diamonds.csv'):
    path = kagglehub.dataset_download(file_directory)
    df = pd.read_csv(path + '/' + file_name).dropna()
    return df[~((df['x'] == 0) | (df['y'] == 0) | (df['z'] == 0) | (df['price'] <= 0))]

def prepare_data(df: pd.DataFrame, scaler: StandardScaler):
    cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    cut_encoder = OrdinalEncoder(categories=[cut_order])
    color_encoder = OrdinalEncoder(categories=[color_order])
    clarity_encoder = OrdinalEncoder(categories=[clarity_order])

    columns = ['carat','depth','table', 'price','x', 'y', 'z']
    norm_data = scaler.fit_transform(df[columns])
    df_normalized = pd.DataFrame(norm_data, columns=[col + '_norm' for col in columns], index=df.index)
    df = pd.concat([df, df_normalized], axis=1).drop("Unnamed: 0", axis=1)

    df['cut_encoded'] = cut_encoder.fit_transform(df[['cut']])
    df['color_encoded'] = color_encoder.fit_transform(df[['color']])
    df['clarity_encoded'] = clarity_encoder.fit_transform(df[['clarity']])

    outputs = df['price_norm'].to_numpy()
    inputs = np.array(df.drop(['carat', 'cut', 'color', 'clarity','depth', 'price', 'table', 'x',  'y', 'z', 'price_norm'], axis=1).values.tolist())
    return inputs, outputs, df.reset_index(drop=True)

def split_data(x, y, test_size = 0.2):
    return train_test_split(x, y, test_size=test_size)

def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_train = False):
    avg_loss = 0
    if is_train:
        model.train()
    else:
        model.eval()

    for idx, (x,y) in enumerate(dataloader):
        if is_train:
            optimizer.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda').unsqueeze(1)

        output = model(x)
        error = criterion(output, y)

        if is_train:
            error.backward()
            optimizer.step()

        #nn.MSELoss(reduction = 'mean')
        avg_loss += error.detach().item()

    scheduler.step()
    return avg_loss, scheduler.get_last_lr()[0]

class TableDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class BaseModel(nn.Module):
    def __init__(self, input_size = 9, output_size = 1):
        super(BaseModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

n_epochs = 50
batch_size = 250
scaler = StandardScaler()
input,price,df = prepare_data(fetch_data(), scaler)
x_train, x_test, y_train, y_test = split_data(input,price)
train_dataset = TableDataset(x_train, y_train)
test_dataset = TableDataset(x_test, y_test)

base_model = BaseModel().to('cuda')

criterion = nn.MSELoss(reduction='mean')
base_optimizer = optim.Adam(base_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(base_optimizer, step_size = 40, gamma = 0.25)

dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
dataloader_test = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

for epoch in range(n_epochs):
    train_loss, learning_rate = run_epoch(base_model, dataloader_train, base_optimizer, criterion, scheduler, is_train = True)
    print(f'Epoch {epoch + 1} / {n_epochs}, Average loss per batch: {train_loss:.4f}, Learning_Rate: {learning_rate:.4f}')

predictions = np.array([])
base_model.eval()
for i, (x,y) in enumerate(dataloader_test):
    x = x.to('cuda')
    y = y.to('cuda').unsqueeze(1)
    out = base_model(x)
    error = criterion(out, y)
    predictions = np.append(predictions, out.cpu().detach().numpy())

predictions = (predictions * scaler.scale_[3]) + scaler.mean_[3]
prices = (y_test * scaler.scale_[3]) + scaler.mean_[3]
print(f'predictions: {predictions}')
print(f'prices: {prices}\n')
print(f'R^2: {r2_score(prices, predictions)}')
print(f'Root Mean Squared Error: {root_mean_squared_error(prices, predictions)}')

plt.figure(figsize = (10,10))
plt.plot([0,max(predictions)],[0,max(predictions)], color = 'red', linestyle = '--', label = 'Ideal Predictions')
plt.scatter(prices, np.clip(predictions,0,None), alpha = 0.7, color = 'black')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.legend()
plt.show()