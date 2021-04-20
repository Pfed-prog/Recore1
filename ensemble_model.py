import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  PowerTransformer
from sklearn.feature_selection import SelectFromModel
from functools import reduce

class  EnsembleModel:

    def __init__(self):
        self.models = dict()

        clf1 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model1'] = Pipeline(steps=[('scaler_1', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_1', clf1)])     

        clf2 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model2'] = Pipeline(steps=[('scaler_2', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_2', clf2)])

        clf3 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model3'] = Pipeline(steps=[('scaler_3', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_3', clf3)])

        clf4 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model4'] = Pipeline(steps=[('scaler_4', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_4', clf4)])
        
        clf5 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model5'] = Pipeline(steps=[('scaler_5', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_5', clf5)])  

        clf6 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model6'] = Pipeline(steps=[('scaler_6', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_6', clf6)]) 

        clf7 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model7'] = Pipeline(steps=[('scaler_7', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_7', clf7)]) 

        clf8 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model8'] = Pipeline(steps=[('scaler_8', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_8', clf8)]) 

        clf9 = RandomForestRegressor(n_estimators = 400, random_state=1)
        self.models['model9'] = Pipeline(steps=[('scaler_19', PowerTransformer()),('dim_red',SelectFromModel(RandomForestRegressor(n_estimators = 400, random_state=1))),('forest_9', clf9)]) 
        
        
        
    def fit1(self, X, y):
        self.models['model1'].fit(X, y)
    def fit2(self, X, y):
        self.models['model2'].fit(X, y)    
    def fit3(self, X, y):
        self.models['model3'].fit(X, y)
    def fit4(self, X, y):
        self.models['model4'].fit(X, y)
    def fit5(self, X, y):
        self.models['model5'].fit(X, y)
    def fit6(self, X, y):
        self.models['model6'].fit(X, y)
    def fit7(self, X, y):
        self.models['model7'].fit(X, y)
    def fit8(self, X, y):
        self.models['model8'].fit(X, y)
    def fit9(self, X, y):
        self.models['model9'].fit(X, y)



    def _predict(self,model_name, inp_X):
        preds = self.models[model_name].predict(inp_X)
        return preds

    def predict(self,inp_X):
        p1 = pd.DataFrame(self._predict('model1',inp_X)).rename({0:0,1:2, 2:4,3:5,4:7,5:8,6:11,7:12,8:14,9:16,10:17,11:19,12:24,13:35}, axis='columns')
        p2 = pd.DataFrame(self._predict('model2',inp_X)).rename({0:1,1:6}, axis='columns')
        p3 = pd.DataFrame(self._predict('model3',inp_X)).rename({0:3,1:18, 2:20,3:21,4:23}, axis='columns')
        p4 = pd.DataFrame(self._predict('model4',inp_X)).rename({0:9,1:13, 2:15}, axis='columns')
        p5 = pd.DataFrame(self._predict('model5',inp_X)).rename({0:10,1:43}, axis='columns')
        p6 = pd.DataFrame(self._predict('model6',inp_X)).rename({0:22,1:30, 2:31,3:32,4:38,5:39,6:41,7:42,8:44,9:45,10:46,11:47}, axis='columns')
        p7 = pd.DataFrame(self._predict('model7',inp_X)).rename({0:25,1:28, 2:34,3:36}, axis='columns')
        p8 = pd.DataFrame(self._predict('model8',inp_X)).rename({0:26,1:27,2:40}, axis='columns')
        p9 = pd.DataFrame(self._predict('model9',inp_X)).rename({0:29,1:33,2:37, 3:48}, axis='columns')
        data_frames = [p1, p2, p3,p4,p5,p6,p7,p8,p9]
        p = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True), data_frames)
        p = p.reindex(sorted(p.columns), axis=1)
        return p
        