import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import  sklearn.model_selection as model_selection
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,r2_score
from sklearn.model_selection import GridSearchCV
import datetime


df_CGL = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx",sheet_name="Main")
df_CGL_val = pd.read_excel(r"Dataset\CGL Model Data_v2.xlsx",sheet_name="Validation")

df_CGL.rename(columns={"TS":"UTS","ELONGATION":"EL"},inplace=True)
df_CGL_val.rename(columns={"TS":"UTS","ELONGATION":"EL"},inplace=True)

df_CGL=df_CGL.drop(["ACTUAL TDC"],axis=1)
df_CGL_val=df_CGL_val.drop(["ACTUAL TDC"],axis=1)


# df_CGL = pd.get_dummies(df_CGL,
#                         columns=['ACTUAL TDC'],
#                         drop_first=True)


# df_CGL_val = pd.get_dummies(df_CGL_val,
#                         columns=['ACTUAL TDC'],
#                         drop_first=True)

Properties_to_predict=["UTS","YS","EL"]
print(df_CGL)

print(df_CGL_val.shape)
print(df_CGL.shape)



for property in Properties_to_predict:
    print("Building Model for:",property)
    property_no=Properties_to_predict.index(property)

    properties=["UTS","YS","EL"]

    properties.remove(property)
    df=df_CGL
    

    df=df.drop(properties,axis=1)
    df_model=df


    x=df_model.drop(property,axis=1)
    y=df_model[property]

    ####Validation Data ###########

    df_CGL_val=df_CGL_val.drop(properties,axis=1)


    x_val=df_CGL_val.drop(property,axis=1)
    y_val=df_CGL_val[property]

    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)

    print("########## Fitting MLR Model #######")

    mlr=LinearRegression()
    mlr.fit(x_train,y_train)
    y_pred=mlr.predict(x_test)
    y_train_pred=mlr.predict(x_train)
    y_val_pred=mlr.predict(x_val)

    print("R2:",metrics.r2_score(y_test,y_pred))
    print("Train R2:",metrics.r2_score(y_train,y_train_pred))
    print("Train R2:",metrics.r2_score(y_val,y_val_pred))


    r2 = round(r2_score(y_test, y_pred), 2)

    print(" ")
    mae=round(mean_absolute_error(y_test,y_pred),2)
    mape=mean_absolute_percentage_error(y_test,y_pred)*100
    mape=round(mape,2)
    print(mape)

    evaluation_metrics=[mae,mape,r2]
    #df_model_evaluation=pd.DataFrame([evaluation_metrics],column=[property+"_"+"MAE",property+"_"+"mape",property+"_"+"r2"])

    ########## Model For Random Forest ##########

    print("###### Fitting Random Forest ALgorithm")

    model=RandomForestRegressor()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("R2:",metrics.r2_score(y_test,y_pred))
    r2 = round(r2_score(y_test, y_pred), 2)

    print(" ")
    mae=round(mean_absolute_error(y_test,y_pred),2)
    mape=mean_absolute_percentage_error(y_test,y_pred)*100
    mape=round(mape,2)
    print(mape)

    evaluation_metrics=[mae,mape,r2]
   # df_model_evaluation=pd.DataFrame([evaluation_metrics],column=[property+"_"+"MAE",property+"_"+"mape",property+"_"+"r2"])


    ########## Model For Random Forest ##########

    print("###### Fitting Random Forest ALgorithm")







