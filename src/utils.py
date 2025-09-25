import matplotlib.pyplot as plt
import seaborn as sns

def plot_survival_count(df):
    sns.countplot(x='Survived', data=df)
    plt.title('Survival Count')
    plt.show()

def plot_survival_by_gender(df):
    sns.countplot(x='Sex', hue='Survived', data=df)
    plt.title('Survival by Gender')
    plt.show()

def plot_survival_by_pclass(df):
    sns.countplot(x='Pclass', hue='Survived', data=df)
    plt.title('Survival by Pclass')
    plt.show()

def plot_survival_by_famsize(df):
    sns.countplot(x='FamilySize', hue='Survived', data=df)
    plt.title('Survival by FamilySize')
    plt.show()

def plot_survival_by_isalone(df):
    sns.countplot(x='IsAlone', hue='Survived', data=df)
    plt.title('Survival by IsAlone')
    plt.xticks([0,1], ['Not Alone', 'Alone'])
    plt.show()

def plot_age_distribution(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title('Age Distribution')
    plt.show()

def plot_age_survival(df):
    sns.kdeplot(df[df['Survived']==1]['Age'], label='Survived', fill=True)
    sns.kdeplot(df[df['Survived']==0]['Age'], label='Did Not Survive', fill=True)
    plt.title('Age Distribution by Survival')
    plt.legend()
    plt.show()

def plot_fare_distribution(df):
    plt.figure(figsize=(10,6))
    sns.histplot(df['Fare'], bins=30, kde=True)
    plt.title('Fare Distribution')
    plt.show()

def plot_fare_vs_survival(df):
    sns.boxplot(x='Survived', y='Fare', data=df)
    plt.title('Fare vs Survival')
    plt.show()
