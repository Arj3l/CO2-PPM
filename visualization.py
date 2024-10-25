import matplotlib.pyplot as plt
import seaborn as sns

def plot_co2_levels(df):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Date', y='Interpolated', data=df)
    plt.title('CO2 Levels Over Time')
    plt.xlabel('Date')
    plt.ylabel('CO2')
    plt.show()

def plot_co2_by_year(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Year', y='Interpolated', data=df)
    plt.title('CO2 Levels Each Year')
    plt.xlabel('Year')
    plt.ylabel('CO2')
    plt.show()