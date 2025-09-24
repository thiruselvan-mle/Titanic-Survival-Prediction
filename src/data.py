def load_data(path):
    import pandas as pd
    return pd.read_csv(path)

def preprocess_mis_val(df):
    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
    df.drop('Cabin', axis=1, inplace=True)
    return df

def drop_cols(df):
    df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    return df

def feature_engi(df):
    import numpy as np
    df['FamilySize']=df['SibSp']+df['Parch']+1
    df['IsAlone'] = np.where(df['FamilySize']==1, 1, 0)
    return df

def encode_var(df):
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
    return df

def basic_info(df):
    if df is None:
        raise ValueError("Input df is None!")
    df.info()
    print(df.head())
    return df

def save_clean_data(df):
    import os
    if not os.path.exists("D:\Thiru\ML_Projects\Titanic-Survival-Prediction\Data\processed"):
        os.makedirs("D:\Thiru\ML_Projects\Titanic-Survival-Prediction\Data\processed")
    
    df.to_csv("D:\Thiru\ML_Projects\Titanic-Survival-Prediction\Data\processed\cleaned_titanic.csv", index=False)
    print("Cleaned dataset saved to D:\Thiru\ML_Projects\Titanic-Survival-Prediction\Data\processed\cleaned_titanic.csv")
    return df
