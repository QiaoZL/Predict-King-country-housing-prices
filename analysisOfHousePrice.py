import pandas as pd
import numpy as np
import sklearn.linear_model as linearmodels
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm.classes import LinearSVR
from math import log
import warnings
warnings.filterwarnings("ignore")


def getScore(result,cv=10):
    count = 0
    #print(result)
    for score in result:
        count += np.sqrt(-score)
    print(count/cv)
    return

def calAIC(estimator, X, y):
        
    y_hat = estimator.predict(X)
    resid = y - y_hat
    sse = sum(resid**2)
    k = len(X[0])
    AIC= 2*k - 2*log(sse)
    
    return AIC


data = pd.read_csv("../datasets/housesalesprediction/kc_house_data.csv")
print(data.columns)
print(data.head(5))

y = data["price"]
print(y.head(5))
print(np.std(y))

features = data.iloc[:,3:]
print(features.head(5))
featurenames = features.columns
#features.drop('zipcode',1,inplace=True)
#features.drop('lat',1,inplace=True)
#features.drop('long',1,inplace=True)

scalerNorm = Normalizer(norm='l2')
scalerStandard = StandardScaler().fit(features)
#scalerX.fit(features)
#features = scalerX.transform(features)
features = scalerStandard.transform(features)

print(features.shape)


Lars_cv = linearmodels.LarsCV(cv=6).fit(features, y)
Lasso_cv = linearmodels.LassoCV(cv=6).fit(features, y)
alphas = np.linspace(Lars_cv.alphas_[0], .1 * Lars_cv.alphas_[0], 6)
Randomized_lasso = linearmodels.RandomizedLasso(alpha=alphas,random_state=42)

linear_regression = linearmodels.LinearRegression()
linear_SVR = LinearSVR(loss='squared_epsilon_insensitive')


featureselector_Lars = feature_selection.SelectFromModel(Lars_cv,prefit=True)
featureselector_Lasso = feature_selection.SelectFromModel(Lasso_cv,prefit=True)
featureselector_RLasso = Randomized_lasso.fit(features,y)

print(Lars_cv.coef_)
print(Lasso_cv.coef_)
print(Randomized_lasso.scores_)

scoreoffeature = pd.DataFrame([Lars_cv.coef_,Lasso_cv.coef_,Randomized_lasso.scores_]
                              , columns= featurenames
                              , index = ['Lars','Lasso','Randomized_lasso'])

scoreoffeature.to_csv( "../datasets/housesalesprediction/feature_weight.csv")
print (scoreoffeature)

X_Lars = featureselector_Lars.transform(features)
X_Lasso = featureselector_Lasso.transform(features)
X_randomLasso = featureselector_RLasso.transform(features)
print(X_Lars.shape)
print(X_Lasso.shape)
print(X_randomLasso.shape)

print("origin features:")

getScore(cross_val_score(Lars_cv, features,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(Lasso_cv, features,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(linear_regression, features,y,cv = 5,scoring = 'mean_squared_error'),5)

print("origin AIC")
print(sum(cross_val_score(Lars_cv, features,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(Lasso_cv, features,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(linear_regression, features,y,cv = 5,scoring = calAIC))/5)

print("selected features, Lars:")

getScore(cross_val_score(Lars_cv, X_Lars,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(Lars_cv, X_Lasso,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(Lars_cv, X_randomLasso,y,cv = 5,scoring = 'mean_squared_error'),5)

print("selected features, Lars AIC")
print(sum(cross_val_score(Lars_cv, X_Lars,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(Lars_cv, X_Lasso,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(Lars_cv, X_randomLasso,y,cv = 5,scoring = calAIC))/5)

print("selected features, Lasso:")

getScore(cross_val_score(Lasso_cv, X_Lars,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(Lasso_cv, X_Lasso,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(Lasso_cv, X_randomLasso,y,cv = 5,scoring = 'mean_squared_error'),5)

print("selected features, Lasso AIC")
print(sum(cross_val_score(Lasso_cv, X_Lars,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(Lasso_cv, X_Lasso,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(Lasso_cv, X_randomLasso,y,cv = 5,scoring = calAIC))/5)

print("selected features, linear regression:")

getScore(cross_val_score(linear_regression, X_Lars,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(linear_regression, X_Lasso,y,cv = 5,scoring = 'mean_squared_error'),5)
getScore(cross_val_score(linear_regression, X_randomLasso,y,cv = 5,scoring = 'mean_squared_error'),5)

print("selected features, linear regression AIC")
print(sum(cross_val_score(linear_regression, X_Lars,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(linear_regression, X_Lasso,y,cv = 5,scoring = calAIC))/5)
print(sum(cross_val_score(linear_regression, X_randomLasso,y,cv = 5,scoring = calAIC))/5)
