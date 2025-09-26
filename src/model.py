import joblib
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve,auc
from sklearn.model_selection import cross_val_score, StratifiedKFold

def separate_x_y(df):
    X=df.drop('Survived',axis=1)
    Y=df['Survived']
    print('Features Columns:',X.columns)
    print('\n')
    print('Target Column','Survived')
    return X, Y

def split_train_test(X, Y, test_size=.2, random_state=42):
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size = test_size, random_state = random_state)
    print('Training Samples', X_train.shape[0])
    print('Testing Samples', X_test.shape[0])
    return X_train, X_test, Y_train, Y_test

def train_ml_model(X_train, Y_train):
    models=[]
    models.append(('LR',LogisticRegression(max_iter=500)))
    models.append(('DT',DecisionTreeClassifier(random_state=42)))
    models.append(('RFC',RandomForestClassifier(random_state=42)))
    models.append(('GBC',GradientBoostingClassifier(random_state=42)))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('SVC',SVC()))
    models.append(('LDA',LinearDiscriminantAnalysis()))
    models.append(('NB',GaussianNB()))

    results=[]
    names=[]
    res=[]

    for name,model in models:
        kfold=StratifiedKFold(n_splits=10, random_state=None)
        cv_results=cross_val_score(model,X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        res.append(cv_results.mean())
        print('%s: %f%%'%(name,cv_results.mean()*100))
    return names, res, results 

def plot_accu_bar(names, res):
    plt.figure(figsize=(10,5))
    plt.bar(names,res,color='skyblue',width=0.6)
    plt.title('Algorithm Comparison')
    plt.xlabel('Model Names')
    plt.ylabel('Accuracy')
    plt.ylim(.750,.830)
    plt.show()

def plot_accu_box(results, names):
    plt.figure(figsize=(10,5))
    plt.boxplot(results,tick_labels=names)
    plt.title('Algorithm Comparison (10-Fold CV)')
    plt.ylabel('Accuracy')
    plt.show()

def accu_table(names, res):
    results_df=pd.DataFrame({
    'Models': names,
    'Mean_Accuracy': [round(r*100,2) for r in res]
    }).sort_values(by='Mean_Accuracy',ascending=False)

    print(results_df)

def separate_X_Y(df):
    X=df.drop('Survived',axis=1)
    Y=df['Survived']
    return X, Y

def shape_x(X, Y):
    X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=.2, random_state=42)

    print('Training Samples', X_train.shape)
    print('Testing Samples', X_test.shape)
    return X_train, X_test, Y_train, Y_test

def final_model(X_train, Y_train, X_test, random_state=42):
    model=RandomForestClassifier(random_state = random_state)
    model.fit(X_train, Y_train)

    Y_pred=model.predict(X_test)
    Y_prob=model.predict_proba(X_test)[:,-1]
    return Y_pred, Y_prob, model

def accuracy(Y_pred, Y_test):
    accu=accuracy_score(Y_pred,Y_test)*100
    print(f'Accuracy_Score:{accu:.2f}')
    return accu

def plot_confusion_matrix(Y_pred,Y_test):
    cm=confusion_matrix(Y_pred,Y_test)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Random Forest Confusion Matrix')
    plt.show()

def clss_report(Y_test, Y_pred):
    print(classification_report(Y_test, Y_pred))

def roc_auc(Y_test, Y_prob):
    fpr, tpr, threshold=roc_curve(Y_test, Y_prob)
    roc_auc=auc(fpr,tpr)

    plt.plot(fpr, tpr,label=f"Roc_Auc(Auc={roc_auc:.2f})",linewidth=2,color='orange')
    plt.plot([0,1],[0,1],color='navy',ls='--')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

def imp_features(X, model):
    feat_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    sns.barplot(x="Importance", y="Feature", data=feat_importances,color='skyblue')
    plt.title("Feature Importance - Random Forest")
    plt.show()

    display(feat_importances)

def accu_auc_table(accu, roc_auc):
    summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC"],
    "Score": [round(accu,2), round(roc_auc,2)]
    })
    display(summary_df)

def kaggle_subm_set(path,model):
    # Load Kaggle test.csv
    test_df=pd.read_csv(path)

    # Preprocessing function
    def preprocess_test(df):
        df['Age']=test_df['Age'].fillna(df['Age'].median())
        df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
        df=test_df.drop('Cabin', axis=1)
        df['Sex'] = df['Sex'].map({'male':0, 'female':1})
        df['Embarked'] = df['Embarked'].map({'C':0, 'Q':1, 'S':2})
        df['FamilySize']=df['SibSp']+df['Parch']+1
        df['IsAlone'] = np.where(df['FamilySize']==1, 1, 0)
        df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
        return df

    #Preprocess Kaggle test set
    test_processed=preprocess_test(test_df)
    
    #Predict using random Forest
    test_prediction=model.predict(test_processed)

    #Prepare Submission
    submission=pd.DataFrame({
        'PassengerId':test_df['PassengerId'],
        'Survived':test_prediction
    })

    import os
    if not os.path.exists("D:\Thiru\ML_Projects\Titanic-Survival-Prediction\Submission.csv"):
    
        submission.to_csv("D:\Thiru\ML_Projects\Titanic-Survival-Prediction\Submission.csv",index=False)
        print("Submission file created: submission.csv")

def save_model(model,filename="D:\Thiru\ML_Projects\Titanic-Survival-Prediction\models\model.pkl"):
    joblib.dump(model, filename)
    print(f'model saved to {filename}')

def load_model(filename="D:\Thiru\ML_Projects\Titanic-Survival-Prediction\models\model.pkl"):
    model=joblib.load(filename)
    print(f'model loaded from {filename}')
    return model