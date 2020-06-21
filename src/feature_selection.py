import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


FEATURES_CLINICA_HISTORIA_PREVIA = [
       'Cancer', 'HAS', 'DM2', 'Cardiopatia Outra',
       'Marcapasso', 'Sincope', 'Fibrilação/Flutter Atrial', 'I R Crônica',
       'DLP', 'Coronariopatia', 'Embolia Pulmonar', 'Ins Cardiaca ', 'AVC',
       'DVP', 'TSH', 'Tabagismo', 'Alcoolismo', 'Sedentarismo']

FEATURES_ELETROCARDIOGRAMA = ['ECG ', 'FC',
       'Alt Prim', 'Dist Cond InterVent ', 'Dist Cond AtrioVent ',
       'Pausa > 3s ', 'ESV', 'EV', 'TVMNS', 'Area Elet inativa'] 

class Correlation():

    def __init__(self, thresh=0.2):
        self.tresh  = thresh
        self.relevant_features = []

    def __check_inputs(self, X, y):
        pass

    def fit(self, X, y):
        '''
        Inputs:
        -------
        X: a dataframe
        y: a series
        '''
        # Stack X and y
        data = pd.concat([X, y], axis=1)
        # X_new = np.append(X.values, y.values.reshape(-1,1), axis=1)
        # y_new = X_new[:, -1]

        # Compute the correlation matrix
        corr_matrix = data.corr()
        corr_target = abs(corr_matrix.iloc[:,-1])

        # Select upper triangle of correlation matrix
        upper = corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find index of feature columns with correlation greater than 0.95
        # to_drop = [column for column in upper.columns if any(upper[column] > self.tresh)]

        # Selecting highly correlated features
        relevant_features = corr_target[corr_target > self.tresh]

        self.scores = relevant_features
        self.relevant_features = relevant_features.index.values

        pass


class BackwardElimination():

    def __init__(self, pvalue=0.05):
        self.pvalue  = pvalue
        self.relevant_features = []

    def __check_inputs(self, X, y):
        pass

    def fit(self, X, y):
        '''
        Inputs:
        -------
        X: a dataframe
        y: a series
        '''
        cols = list(X.columns)
        pmax = 1
        while (len(cols)>0):
            p = []
            X_1 = X[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(y, X_1).fit()
            p = pd.Series(model.pvalues.values[1:],index = cols)      
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if(pmax > self.pvalue):
                cols.remove(feature_with_p_max)
            else:
                break
        selected_features_BE = cols

        self.relevant_features = selected_features_BE

        pass


class RecursiveFeatureElimination():

    def __init__(self, test_size=0.3, random_state=42):
        self.test_size  = test_size
        self.random_state = random_state
        self.relevant_features = []


    def __check_inputs(self, X, y):
        pass

    def fit(self, X, y):
        '''
        Inputs:
        -------
        X: a dataframe
        y: a series
        '''
        # model = LinearRegression()
        # #Initializing RFE model
        # rfe = RFE(model, 7)
        # #Transforming data using RFE
        # X_rfe = rfe.fit_transform(X, y)  
        # #Fitting the data to model
        # model.fit(X_rfe,y)

        # no of features
        nof_list=np.arange(1, X.shape[1])            
        high_score = 0
        #Variable to store the optimum features
        nof = 0           
        score_list = []

        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=self.test_size)
            model = LinearRegression()
            rfe = RFE(model,nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]

        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))

        cols = list(X.columns)
        model = LinearRegression()
        #Initializing RFE model
        rfe = RFE(model, nof)             
        #Transforming data using RFE
        X_rfe = rfe.fit_transform(X,y)  
        #Fitting the data to model
        model.fit(X_rfe,y)              
        temp = pd.Series(rfe.support_,index = cols)
        selected_features_rfe = temp[temp==True].index
        
        self.relevant_features = selected_features_rfe.values

        pass


class ANOVA():

    def __init__(self, percentile=0.05):
        self.percentile = percentile
        self.relevant_features = []


    def __check_inputs(self, X, y):
        pass

    def fit(self, X, y):
        '''
        Inputs:
        -------
        X: a dataframe
        y: a series
        '''
        relevance = SelectPercentile(f_classif, percentile=self.percentile)
        feature_relevant = relevance.fit_transform(X, y.values.ravel())

        idx_most_relevant = relevance.get_support()
        names_most_relevant = X.columns[idx_most_relevant]

        scores = -np.log10(relevance.pvalues_[idx_most_relevant])
        scores /= scores.max()

        self.scores = scores
        self.relevant_features = names_most_relevant

        pass



class Embedded():

    def __init__(self, percentile=0.05):
        self.percentile = percentile
        self.relevant_features = []


    def __check_inputs(self, X, y):
        pass

    def fit(self, X, y):
        '''
        Inputs:
        -------
        X: a dataframe
        y: a series
        '''
        reg = LassoCV()
        reg.fit(X, y)
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        print("Best score using built-in LassoCV: %f" %reg.score(X,y))
        coef = pd.Series(reg.coef_, index = X.columns)

        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        # If the feature is irrelevant, lasso penalizes it’s coefficient and make it 0. 
        # Hence the features with coefficient = 0 are removed and the rest are taken
        relevant_features = coef[coef != 0]

        self.scores = coef.sort_values()
        self.relevant_features = relevant_features.index.values

        pass


class Ensemble():

    def __init__(self, thresh=0.05, goal='classification'):
        self.thresh = thresh
        self.relevant_features = []
        self.goal = goal


    def __check_inputs(self, X, y):
        pass

    def fit(self, X, y):
        '''
        Inputs:
        -------
        X: a dataframe
        y: a series
        '''
        # Build a forest and compute the feature importances
        if self.goal == 'regression':
            forest = RandomFortestRegressor()
        else:
            forest = RandomForestClassifier(n_estimators=250,
                                        random_state=0)

        forest.fit(X, y)

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]
        feature_names = X.columns

        self.scores = importances[indices]
        self.feature_names = feature_names[indices]
        self.std = std[indices]
        self.indices = indices
        
        # Select the relvante by a predefined threshold
        relevant_importances = self.scores[self.scores >= self.thresh]
        self.relevant_features = self.feature_names[self.scores >= self.thresh]
        
        

        pass



if __name__ == "__main__":
    from os import path

    DATA_RAW_FOLDER = '../../data/ECG_Chagasicos_HUCFF_UFRJ/'
    DATA_FILENAME = 'Clinical_data_09-09-19-processed.csv'

    data_clinical = pd.read_csv(path.join(DATA_RAW_FOLDER,DATA_FILENAME))

    X = data_clinical[FEATURES_ELETROCARDIOGRAMA]
    y = data_clinical["Obito"]
    
    # Correlation
    fs = Correlation()
    fs.fit(X, y)
    print("By correlation: {}".format(fs.relevant_features))
    
    # Backward Elimination (BE)
    fs = BackwardElimination()
    fs.fit(X, y)
    print("By Backward Elimination: {}".format(fs.relevant_features))

    # Recursive Feature Elimination (RFE)
    fs = RecursiveFeatureElimination()
    fs.fit(X, y)
    print("By Recursive Feature Elimination: {}".format(fs.relevant_features))

    # ANOVA
    fs = ANOVA()
    fs.fit(X, y)
    print("By Anova: {}".format(fs.relevant_features))

    # Embedded
    fs = Embedded()
    fs.fit(X, y)
    print("By Embedded: {}".format(fs.relevant_features))

    # Ensemble
    fs = Ensemble()
    fs.fit(X, y)
    print("By Ensemble: {}".format(fs.relevant_features))






