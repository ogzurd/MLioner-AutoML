import streamlit as st 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

 
#sadece sınıflandırıcılar için olsun.

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,mean_squared_error,plot_confusion_matrix,confusion_matrix,classification_report
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(
        page_title="MLioner",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )




st.sidebar.write("""# Welcome to Mlioner
**Mlioner** will help you for built your machine learning models easily.""")
#file upload as a csv or excel 


file = st.sidebar.file_uploader("Choose a CSV or Excel File", type=["csv","xlsx"])




#get dataset function
def get_dataset():
    #if file was selected
    if file is not None:

        #seperate dataset by ",", or ";" 
        sep = st.sidebar.radio("Select seperator for DataFrame", ("None",";",","),) 
        st.write("## DataFrame")
        if sep == "None":
            df = pd.read_csv(file)
        elif sep==";":
            df = pd.read_csv(file, sep=";")
        elif sep == ",":
            df = pd.read_csv(file, sep=",")

        return df
    
    else:
        st.write("## Please choose a CSV file.")
#get dataset
df = get_dataset() 
if df is not None:
    st.write(df)
    st.write("Shape of DataFrame :",df.shape)
    if st.checkbox("Show describe of DataFrame"):
        if st.checkbox("T"):
            des = st.write(df.describe().T)
        else:
            des = st.write(df.describe())
    if st.checkbox("See Correlation Map"):
        plt.style.use("dark_background")
        fig,ax = plt.subplots(figsize = (16,10))
        sns.heatmap(df.corr(),annot=True,cmap="Blues")
        st.write(fig)


#drop columns function
def drop_columns():

    #if df was selected
    if df is not None:
        st.write("### Drop Columns")
        #drop features vertically
        drop_columns = st.multiselect("If you want to drop the features, you can select. If you don't, you can skip this section.", options=df.columns)
        
        if drop_columns:
            df_drop = df.drop(drop_columns, axis=1)
            return df_drop
        else:
            return df
#get df with dropped columns
if file is not None:
    df  = drop_columns()
    st.write(df.head(3))


#drop null values function
def null_value():

    if st.checkbox("Show Null Values"):
        st.write("Sum of null values : ",df.isnull().sum().sum())
        sumofnan = df.isnull().sum().sum()
        st.write("Null values : ",df.isnull().sum())
        if sumofnan > 0 :
            if st.button("Drop Null Values"):
                df_n = df.dropna()
                return df_n
            else:
                return df
        else:
            st.write("There are no null values.")
            return df
    else:
        return df 
#get df with dropped value
if file is not None:
    df = null_value()
    st.write("Shape of DataFrame mood :",df.shape)






#X,y seperate function
def X_and_Y():
   
    if file is not None:
        
        st.write("### Select Target Value")

        target_value = st.selectbox("What is your y value?", options=df.columns)
    
        X  = df.drop(df[[target_value]],axis=1)
        y = df[[target_value]]

        scaling = st.radio("Do you want to Min-Max Scaling?", ("Yes","No"))

        if scaling == "Yes":
            X = (X-X.min())  / (X.max()-X.min())
 
        return X,y  
#get X,y that are seperate
if df is not None:
    try:
        X,y = X_and_Y() 
        st.write("Your target value",y.head(3))

        value_c = st.checkbox("Show value counts")
        if value_c:
            st.write(y.value_counts())      
        
        st.write("Your data",X.head(3))
    except:
        st.warning("Please select your target value correctly..")



#balanced data with smote
def balance_data():
    if file is not None:
        st.write("### Balanced Data with SMOTE")
        st.write("**If you don't want to balance dataset, skip unchecking box below.**")

        try:
            X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=.2, random_state=42)
            if st.checkbox("I want to balance data with SMOTE"):
                k_neighbors = st.slider("k_neighobors", 1,10)
            
                sm = SMOTE(random_state=42,k_neighbors=k_neighbors)
                X_res, y_res = sm.fit_resample(X_train, y_train)
                
                if st.checkbox("Show value counts after SMOTE"):
                    st.write(y_res.value_counts())
                return X_res, y_res 
            else:
                 return X,y
                
        except:
            st.write("*Please update k_neighbors value.*")
#get balance data
if file is not None:
 
    
    try:
        X,y = balance_data()
        st.write("Shape : ",X.shape)
    except:
        st.write("***")


#get data  +++
#drop columns +++
#drop null value +++
#seperate X and y +++
#train-test spliting +++
#balance data +++


#Classifier selectbox
if file is not None:
    classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Random Forest Classifier","MLPClassifier","KNeighbor Classifier"))


#get params of classifiers
def params_classifier():

    if file is not None:
        params = dict()
        if classifier == "Logistic Regression":
            penalty = st.sidebar.radio("penalty",("l2","l1","elasticnet","none"))
            C = st.sidebar.slider("C",1.0,100.0)
            solver = st.sidebar.radio("solver",("lbfgs","newton-cg","liblinear","sag","saga"))

            params["penalty"] = penalty
            params["C"] = C
            params["solver"] = solver
        elif classifier == "Random Forest Classifier":
            n_estimators = st.sidebar.slider("n_estimators", 50,5000)
            min_samples_split = st.sidebar.slider("min_samples_split", 1,25)
            min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1,25)

            params["n_estimators"] = n_estimators
            params["min_samples_split"] = min_samples_split
            params["min_samples_leaf"] = min_samples_leaf
        elif classifier == "MLPClassifier":
            hidden_layer_sizes = st.sidebar.slider("hidden_layer_sizes", value=(50,1000))
            activation = st.sidebar.radio("activation",("identity","logistic","tanh","relu"))
            solver = st.sidebar.radio("solver",("lbfgs","sgd","adam"))
            alpha = st.sidebar.slider("alpha", 0.0001,2.0,00.1)
            
            params["hidden_layer_sizes"] = hidden_layer_sizes
            params["activation"] = activation
            params["solver"] = solver
            params["alpha"] = alpha

        elif classifier == "KNeighbor Classifier":
            n_neighbors = st.sidebar.slider("n_neighbors",3,20)
            weights = st.sidebar.radio("weights",("uniform","distance"))
            algorithm = st.sidebar.radio("algorithm",("auto","ball_tree","kd_tree","brute"))
            p = st.sidebar.slider("p",1,20)

            params["n_neighbors"] = n_neighbors
            params["weights"] = weights
            params["algorithm"] = algorithm
            params["p"] = p

        return params
#get params
if file is not None:
    params = params_classifier()
#define classifier
def get_classifier():
   
    if file is not None:
        if classifier == "MLPClassifier":
            clf = MLPClassifier(hidden_layer_sizes=params["hidden_layer_sizes"], activation=params["activation"]
            ,solver=params["solver"], alpha=params["alpha"])
        elif classifier == "Logistic Regression":
            clf = LogisticRegression(penalty=params["penalty"], C=params["C"]
            ,solver=params["solver"])
        elif classifier == "Random Forest Classifier":
            clf = RandomForestClassifier(n_estimators=params["n_estimators"], min_samples_split=params["min_samples_split"]
            ,min_samples_leaf=params["min_samples_leaf"])
        elif classifier == "KNeighbor Classifier":
            clf = KNeighborsClassifier(n_neighbors = params["n_neighbors"],weights = params["weights"]
            ,algorithm = params["algorithm"], p = params["p"])
        return clf



#get model
def get_models():

    if file is not None:
        clf = get_classifier()
        st.write("### Train-Test Split for Scores")
        tsize = st.slider("Test Size",0.1,0.99)
        
        X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = tsize,random_state=42)
        st.write("Train Shape :",X_train.shape)
        st.write("Test Shape :",X_test.shape)
        try:
            
            model = clf.fit(X_train, y_train)
            model_te = clf.fit(X_test,y_test)
            y_pred = model.predict(X_test)
            y_pred_te = model_te.predict(X_test)
            st.success("*Your model is trained successfully..*")
            return X_train,X_test,y_train,y_test,y_pred,y_pred_te
            
             
                
        except:
            st.write("### WARNING")
            st.write("*This warning can have multiple reasons. We suggest you fix the following causes.*")
            st.write("***")
            st.write("""##### 1- Some hyperparameters can not work together. Please update your hyperparameter values.""")
            st.write("""##### 2- Update your k_neighbors.""")
            st.write("""##### 3- Check your target value.""")
        



    
#get scores
if file is not None:
    try:

        
        X_train,X_test,y_train,y_test,y_pred,y_pred_te = get_models()
        clf = get_classifier()
        st.write("## Scores")
        
        rd = st.radio("Metrics",("Accuracy Score","Cross Validation Score","Confusion Matrix"))

        if rd == "Accuracy Score":
            st.write("#### Accuracy Score of Train :",accuracy_score(y_test, y_pred))
            st.write("#### Accuracy Score of Test :",accuracy_score(y_test, y_pred_te))
        elif rd == "Cross Validation Score":
            cv = st.slider("CV",2,15)
            if cv:
                cvs = cross_val_score(clf, X=X,y=y,cv=cv)
                st.write("### Cross Validation Scores",cvs)
                if st.checkbox("Mean"):
                    st.write("#### Mean of Cross Validation Score :",cvs.mean())
                if st.checkbox("Std"):
                    st.write("#### Standard Deviation of Cross Validation Score :",cvs.std())
        elif rd == "Confusion Matrix":
            
            col1, col2 = st.columns([3,1])

            with col1:
                report = classification_report(y_test,y_pred, output_dict=True)
                clsrep = pd.DataFrame(report).transpose()
                st.write(clsrep)
                
            col2.write(confusion_matrix(y_test,y_pred))
            
            
             


                
                
    except:
        st.write("***")








    
 #