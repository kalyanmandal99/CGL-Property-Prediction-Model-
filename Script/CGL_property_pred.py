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


df_CGL = pd.read_excel(r"Dataset\CGL Model Data.xlsx")
df_CGL.rename(columns={"TS":"UTS","ELONGATION":"EL"},inplace=True)


Properties_to_predict=["UTS","YS","EL"]

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

    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=0)

    print("########## Fitting MLR Model #######")

    mlr=LinearRegression()
    mlr.fit(x_train,y_train)
    y_pred=mlr.predict(x_test)
    print("R2:",metrics.r2_score(y_test,y_pred))
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

    print("###### Fitting Random Forest ALgorithm")c







