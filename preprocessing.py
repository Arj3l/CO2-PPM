import pandas as pd

#Load the dataset
def load_data():
    CO2_PPM = './CSV/CO2.csv'
    df = pd.read_csv(CO2_PPM)
    return df

def clean_data(df):
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

def preprocess_data(df):
    X = df[['Year', 'Month']]
    y = df['Interpolated']
    return X, y
