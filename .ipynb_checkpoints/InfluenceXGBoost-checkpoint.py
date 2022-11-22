import numpy as np

import random
import gc
from scipy import stats
from sklearn.metrics import f1_score

# 分類時の負例は０
class OriginalXgBoostDecisoinTree:
    def __init__(self, log_odds, max_depth,class_weight=None, min_sample=1,
    feature_select='all',lamda=1,imbalanced_weight=False, focal_loss=False):
        self.log_odds = log_odds
        self.max_depth = max_depth
        self.tree = None
        self.class_weight= class_weight
        self.min_sample=min_sample
        self.feature_select = feature_select
        self.lamda = lamda
        self.imbalanced_weight=imbalanced_weight
        self.focal_loss=focal_loss
        self.leaf_num = 0
    def fit(self, data, target):
        self.type = 'clasiffication' if len(np.unique(target)) < 3 else 'regression'
        initial_depth = 1
        
        self.tree = Node_XgBoost(data, target,log_odds = self.log_odds, lamda = self.lamda, max_depth = self.max_depth, 
            class_weight=self.class_weight,min_sample=self.min_sample,feature_select = self.feature_select,
            imbalanced_weight=self.imbalanced_weight, focal_loss=self.focal_loss
            )
        self.tree.split(initial_depth)
    def predict(self, data):
        pred = []
        for s in data:
            pred.append(self.tree.predict(s)) # tree.predictは木の末端のノードを返す
        return np.array(pred)


class InfluencedXgBoost:
    """
        only binay classification
    """
    def __init__(self, max_depth,class_weight=None, min_sample=10, feature_select='all',reg_lambda=1,
                n_estimator = 10,learning_rate = 1, imbalanced_weight=False, switch_turn=0.5,focal_loss=False):
        self.max_depth = max_depth
        self.tree = None
        self.class_weight= class_weight
        self.min_sample=min_sample
        self.feature_select = feature_select
        self.lamda = reg_lambda
        self.n_estimator = n_estimator
        self.learning_rate = learning_rate
        self.models = []
        self.imbalanced_weight = imbalanced_weight
        self.swicth_turn = switch_turn
        self.focal_loss = focal_loss
        

    def fit(self,X, y):
        from imblearn.under_sampling import RandomUnderSampler
        prob = y.mean() 
        
        f0 = np.ones(len(y)) * np.log(prob/(1-prob+0.001))
        swicth = int(self.n_estimator*self.swicth_turn)
        for i in range(self.n_estimator):
                
            if i < swicth:
                # not IB weight
                model = OriginalXgBoostDecisoinTree(log_odds=f0, max_depth=self.max_depth,lamda=self.lamda,
                    min_sample=self.min_sample, feature_select=self.feature_select,class_weight=self.class_weight,
                    imbalanced_weight=False, focal_loss=self.focal_loss)
            else:
                model = OriginalXgBoostDecisoinTree(log_odds=f0, max_depth=self.max_depth,lamda=self.lamda,
                    min_sample=self.min_sample, feature_select=self.feature_select,class_weight=self.class_weight,
                    imbalanced_weight=self.imbalanced_weight, focal_loss=self.focal_loss)


            p = 1/(1+np.exp(-f0))
            # l = y*(np.log(p) ) +(1-y)*(np.log(1-p)) # 
            
            model.fit(X, y)
        
            self.models.append(model)
            f0 += self.learning_rate * model.predict(X)
                        

    def predict_log_odds(self,X):
        
        return np.array( [ m.predict(X) for m in self.models] ).sum(axis=0)

    def predict_proba(self, X):
        log_odds = self.predict_log_odds(X)
        return 1/(1+np.exp(-log_odds))

    def predict(self,X):
        answer = self.predict_proba(X)
        return np.round(answer)

    def calc_f1_score(self, X,y):
        pred = self.predict(X)
        return f1_score(y_true=y,y_pred=pred)

    
    
    
class Node_XgBoost:
    def __init__(self, X, y,log_odds,lamda, max_depth:int, position:str = None,
                    class_weight:dict = None, min_sample:int = 2,feature_select = 'all',sample_weight=None,
                    imbalanced_weight= False,focal_loss=False):
        self.__X = X
        self.__y = y
        self.__log_odds = log_odds
        
        self.__max_depth = max_depth
        # self.__label = np.argmax(np.bincount(y))
        self.__lamda = lamda
        self.__left = None #left child node
        self.__right = None #right child node
        self.__depth = 1
        self.__feature = None
        self.__threshold = None
        self.__gain_max = None
        self.__position = position #if I'm top node , the value is None
        self.__class_weight = class_weight
        self.__min_sample = min_sample
        self.__feature_select = feature_select
        self.__sample_weight = sample_weight
        self.__imbalanced_weight = imbalanced_weight
        self.__focal_loss=focal_loss

        self.set_node_weight()


    def set_node_weight(self):
        
        prob = 1/(1+ np.exp(-self.__log_odds))

        if self.__sample_weight is not None:
            weight0 = self.__sample_weight
        else:
            weight0 = np.ones(len(self.__y))


        if self.__focal_loss:
            weight0 = (self.__y*(1-prob) + (1- self.__y)*prob)**0.5

        class0 = 1
        class1 = 1        
        if self.__class_weight is not None:
            class0 = self.__class_weight[0]
            class1 = self.__class_weight[1]

        if not self.__imbalanced_weight:
            g = ((self.__y - prob)*weight0).sum()
            h = ((prob*(1-prob))*weight0).sum() + self.__lamda
            result = 0.5* (g/h)
            self.__node_weight = result
            self.__node_loss = self.calc_loss(self.__X,self.__y,self.__log_odds)
        else:
            n_0 = len(self.__y[self.__y == 0])
            n_1 = len(self.__y[self.__y == 1])
            prob1 = prob[self.__y == 1]
            prob0 = prob[self.__y == 0]
            result = (n_1 - n_0) / (prob1.sum() + (1-prob0).sum() + 2*self.__lamda)
            # ========================
            self.__node_weight = result
            self.__node_loss = self.calc_loss(self.__X,self.__y,self.__log_odds)


    def calc_loss(self,X,y,log_odds, sample_weight=None):
        prob = 1/(1+ np.exp(-log_odds))

        weight0 = np.ones(len(y))
        class0 = 1
        class1 = 1
        
        
        if self.__class_weight is not None:
            weight0 *= y*(self.__class_weight[1]) + (1-y)*(self.__class_weight[0])
            class0 = self.__class_weight[0]
            class1 = self.__class_weight[1]
            
        if sample_weight is not None:
            weight0 *= sample_weight

        if self.__focal_loss:
            weight0 = (y*(1-prob) + (1- y)*prob)**0.5

        weight = weight0

        if self.__imbalanced_weight:

            n_0 = len(y[y == 0])
            n_1 = len(y[y == 1])

            prob0 = prob[y == 0]
            prob1 = prob[y == 1]

            bunshi = (n_1 - n_0)**2
            bunbo = 2 * (prob1.sum() + (1-prob0).sum() + 2*self.__lamda)
            
            result = -bunshi / bunbo
            # ==================
            return result
        else:
            # ===== L0 =======
            # ================
            
            g = ((y - prob)*weight).sum()
            h = ((prob*(1-prob))*weight).sum() + self.__lamda
            tmp = g**2
            return -0.25 * (tmp/h)
            
            

    def left(self):
        return self.__left
    def right(self):
        return self.__right
        
    def node_weight(self):
        return self.__node_weight

    def log_odds(self):
        return self.__log_odds

    def feature(self):
        return self.__feature

    def threshold(self):
        return self.__threshold

    def split_infomation(self):
        return f'now weight {self.__node_weight} .if feature X{self.__feature} >= {self.__threshold}  then go to left else go to right'
        
    def calc_information_gain(self, feat_idx, z_threshold ):
        x_max = max(self.__X[:, feat_idx])
        x_min = min(self.__X[:, feat_idx])
        threshold = z_threshold*(x_max -x_min) + x_min
        now_score = self.__node_loss
        log_odds = self.log_odds()

        over_y, under_y = self.__y[self.__X[:, feat_idx] >= threshold], self.__y[self.__X[:, feat_idx] < threshold]
        if self.__sample_weight is not None:
            over_sample_weight, under_sample_weight = self.__sample_weight[self.__X[:, feat_idx] >= threshold], self.__sample_weight[self.__X[:, feat_idx] < threshold]
        else:
            over_sample_weight, under_sample_weight = None, None

        over_log_odds  = log_odds[self.__X[:, feat_idx] >= threshold]
        under_log_odds = log_odds[self.__X[:, feat_idx] < threshold]


        over_X = self.__X[self.__X[:, feat_idx] >= threshold, :]
        under_X = self.__X[self.__X[:, feat_idx] < threshold, :]
        

        left_size, right_size = len(over_y), len(under_y)
       
        if left_size == 0  or right_size == 0:
            return -np.inf


            
        left_score =  self.calc_loss(over_X, over_y,over_log_odds, over_sample_weight)
        right_score = self.calc_loss(under_X, under_y, under_log_odds, under_sample_weight)
        information_gain = (now_score -left_score - right_score)
        return information_gain


    def search_best_split(self):
        # def search_best_split(data, target, score):   
        features = self.__X.shape[1]
        best_thrs = None
        best_f = None
        gain = None
        
        gain_max = -np.inf
        kouho = self.feature_candidate(features)
        for feat_idx in kouho:
            now_feature = self.__X[:, feat_idx]
            max_value,min_value = max(now_feature),min(now_feature)
            if max_value - min_value < abs(0.001):
                continue
            z = (now_feature - min_value )/(max_value - min_value) #0-1正規化
            f = np.linspace(0.01, 0.99, 20)
            for value in f:
                information_gain = self.calc_information_gain(feat_idx, value)
                if gain_max < information_gain:  # 最大情報利得の更新
                    gain_max = information_gain
                    best_thrs = value * (max_value - min_value) + min_value # しきい値
                    best_f = feat_idx #　特徴量の列番号
        return gain_max, best_thrs, best_f

    
    def check_next_split(self):
        """
            これ以上分岐しなくて良い場合はTrueを返す。まだ分岐しても良い場合は、Falseを返す
        """
        condition1 = self.__depth > self.__max_depth # 今がmax_depthより大きければTrue
        # condition2 = self.__gain_max < 0.001
        condition3 = self.__min_sample >= len(self.__y)
        return condition1 or condition3

    def split(self, depth):
        self.__depth = depth
        if self.check_next_split(): # あまりにも利得の更新幅が小さければもう分割しない
            return None
        else:
            self.__gain_max, self.__threshold, self.__feature = self.search_best_split()
            if self.__gain_max == -np.inf:
                return None
            
            try:
                idx_left = self.__X[:, self.__feature] >= self.__threshold
                idx_right = self.__X[:, self.__feature] < self.__threshold
            except:
                print(self.__gain_max, self.__threshold, self.__feature)

            if self.__sample_weight is not None:
                            
                self.__left = Node_XgBoost(self.__X[idx_left],  self.__y[idx_left], self.__log_odds[idx_left], self.__lamda,
                    self.__max_depth, position="left",class_weight=self.__class_weight,feature_select = self.__feature_select,
                    sample_weight = self.__sample_weight[idx_left],imbalanced_weight=self.__imbalanced_weight,focal_loss=self.__focal_loss)
                self.__right = Node_XgBoost(self.__X[idx_right], self.__y[idx_right], self.__log_odds[idx_right], self.__lamda,
                    self.__max_depth, position="right", class_weight=self.__class_weight,feature_select = self.__feature_select,
                    sample_weight = self.__sample_weight[idx_right],imbalanced_weight=self.__imbalanced_weight,focal_loss=self.__focal_loss)

            else:
                self.__left = Node_XgBoost(self.__X[idx_left],  self.__y[idx_left], self.__log_odds[idx_left], self.__lamda,
                    self.__max_depth, position="left",class_weight=self.__class_weight,feature_select = self.__feature_select,
                    imbalanced_weight=self.__imbalanced_weight,focal_loss=self.__focal_loss)
                self.__right = Node_XgBoost(self.__X[idx_right], self.__y[idx_right], self.__log_odds[idx_right], self.__lamda,
                    self.__max_depth, position="right", class_weight=self.__class_weight,feature_select = self.__feature_select,
                    imbalanced_weight=self.__imbalanced_weight,focal_loss=self.__focal_loss)

            self.__left.split(self.__depth +1)
            self.__right.split(self.__depth +1)
        
    
    def predict(self, data):
        
        if self.__left is None and self.__right is None:
            return self.__node_weight
        else:
            
            if data[self.__feature] >= self.__threshold:
                
                return self.__left.predict(data)
            else:
                
                return self.__right.predict(data)
                
    def feature_candidate(self,n):
        if self.__position == 'left':
            random.seed(10 + self.__depth)
        elif self.__position == 'right':
            random.seed(20 + self.__depth)
        else:
            random.seed(30 + self.__depth)
            
        if self.__feature_select == 'all':
            return range(n)
        elif self.__feature_select == 'sqrt':
            features = random.sample(range(n),int(np.sqrt(n)))
            return features
        else:
            return range(n)
