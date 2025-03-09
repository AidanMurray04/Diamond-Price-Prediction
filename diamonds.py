import kagglehub
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
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

def PCA_graph(inputs, values, pca, is_prediction = False):

    inputs_pca = pca.fit_transform(inputs)
    fig = plt.figure(figsize=(12,12))
    features = ['carat','depth','table','x', 'y', 'z', 'cut', 'color', 'clarity']
    colors = ["red","orange","brown","magenta","gray","black","gold","pink","cyan"]

    print(pca.components_[:])

    plt.scatter(inputs_pca[:, 0], inputs_pca[:, 1], c=values)
    if is_prediction:
        plt.colorbar(label = 'Relative Predicted Price')
    else:
        plt.colorbar(label = 'Relative Price')
    for i, feature in enumerate(features):
        scale = 5
        extra_x = 5
        extra_y = 10

        loading = pca.components_[:, i]
        plt.arrow(0, 0, loading[0] * scale, loading[1] * scale,
                  color=colors[i], width=0.01, head_width=0.1, length_includes_head=True)

        if feature == 'color' or feature == 'clarity' or feature == 'cut':
            extra_x = 0.5
            extra_y = 0.5
        elif feature == 'table':
            extra_y = 1

        plt.text(
            loading[0] * (scale + extra_x),
            loading[1] * (scale + extra_y),
            feature,
            color=colors[i],
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )
    plt.xlabel('PC1 (~Size vs. ~Cut/Clarity)')
    plt.ylabel('PC2 (~Color vs. ~Clarity)')
    plt.show()
    plt.close()


    '''
    center = inputs_pca.mean(axis=0)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    scatter = ax.scatter(inputs_pca[:, 0],
                         inputs_pca[:, 1],
                         inputs_pca[:, 2],
                         c=values, cmap='viridis', alpha=0.6, depthshade = False, zorder=1)
    fig.colorbar(scatter, ax=ax, pad=0.1, label = 'Relative Price')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    for i, feature in enumerate(features):
        loading = pca.components_[:, i]

        ax.quiver(center[0], center[1], center[2],
                  loading[0]*scale,
                  loading[1]*scale,
                  loading[2]*scale,
                  color='red',
                  arrow_length_ratio=0.2,
                  zorder=1000)

        ax.text(loading[0]*(scale+extra),
                loading[1]*(scale+extra),
                loading[2]*(scale+extra),
                s = feature,
                color='red',
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                zorder=100)

    ax.view_init(elev=20, azim=135)
    ax.dist = 1
    fig.show()
    plt.close()
    '''

def run_epoch(model, dataloader, optimizer, criterion, scheduler, is_train = False):
    avg_loss = 0
    if is_train:
        model.train()
    else:
        model.eval()

    for idx, (x,y) in enumerate(dataloader):
        if is_train:
            optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

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


device = 'cpu'
n_epochs = 50
batch_size = 100
scaler = StandardScaler()
pca = PCA(n_components = 3)
input, price, df = prepare_data(fetch_data(), scaler)
x_train, x_test, y_train, y_test = split_data(input,price)
train_dataset = TableDataset(x_train, y_train)
test_dataset = TableDataset(x_test, y_test)

PCA_graph(input, price, pca)

base_model = BaseModel().to(device)

criterion = nn.MSELoss(reduction='mean')
base_optimizer = optim.Adam(base_model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(base_optimizer, step_size = 40, gamma = 0.25)

dataloader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
dataloader_test = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

for epoch in range(n_epochs):
    train_loss, learning_rate = run_epoch(base_model, dataloader_train, base_optimizer, criterion, scheduler, is_train = True)
    #print(f'Epoch {epoch + 1} / {n_epochs}, Average loss per batch: {train_loss:.4f}, Learning_Rate: {learning_rate:.4f}')

predictions = np.array([])
base_model.eval()
for i, (x, y) in enumerate(dataloader_test):
    x = x.to(device)
    y = y.to(device).unsqueeze(1)
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