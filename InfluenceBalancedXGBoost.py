# 分類時の負例は０
import gc
import random
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

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
        self.__left = None #左側の子ノード
        self.__right = None #右側の子ノード
        self.__depth = 1 # このノードの深さ
        self.__feature = None #　選択された特徴量（の列番号）
        self.__threshold = None #  選択されたしきい値
        self.__gain_max = None #このノードにおける最大情報利得
        self.__position = position #自分の親にとって"left"かrightか? TopならばNone
        self.__class_weight = class_weight
        self.__min_sample = min_sample
        self.__feature_select = feature_select
        self.__sample_weight = sample_weight
        self.__imbalanced_weight = imbalanced_weight
        self.__focal_loss=focal_loss

        self.set_node_weight()

    def sigmoid(self, x):
        x[x<-700] = -700
        return 1/(1+ np.exp(-x))
    def set_node_weight(self):
        
        prob = self.sigmoid(self.__log_odds)
        # prob = 1/(1+ np.exp(-self.__log_odds))
        weight0 = np.ones(len(self.__y))

        if self.__class_weight is not None:
          class_weight = self.__class_weight
          weight0 = self.__y*class_weight[1] + ( 1- self.__y)*class_weight[0]
          weight0 = len(weight0)*weight0/sum(weight0)
            

        if self.__focal_loss:
          weight0 = (self.__y)*(1-prob) + (1-self.__y)*(prob)
          weight0 = (1 - weight0) ** self.__focal_loss
          weight0 = len(weight0)*weight0/sum(weight0)
            

        if  self.__imbalanced_weight:
          # weight0 = self.__sample_weight
          ib_bunshi = abs(prob - self.__y)
          ib_bunbo = ((prob*(1-prob))).sum() + self.__lamda
          ib = ib_bunshi /(ib_bunbo + 1e-5)
          weight0 =  1/(ib+1e-5)
          # weight0 = len(weight0)*weight0/sum(weight0)

          if len(weight0) != len(self.__y):
            raise ValueError(f"set_node_weightでのサイズエラー{weight0.shape} {self.__y.shape}")
        
        g = ((prob - self.__y)*weight0).sum()
        h = ((prob*(1-prob))*weight0).sum() + self.__lamda
        result = -g / (h+ 1e-3)
        self.__total_gi = g
        self.__total_hi = ((prob*(1-prob))*weight0).sum()
        
        self.__node_weight = result
        self.__node_loss = self.calc_loss(self.__X, self.__y, self.__log_odds, sample_weight=weight0)
        self.__total_hessian_with_lambda = h
    
    def calc_loss(self,X,y,log_odds, sample_weight=None):
        weight0 = np.ones(len(y))
        # prob = 1/(1 + np.exp(-log_odds))
        prob = self.sigmoid(log_odds)

        if self.__class_weight is not None:
          weight0 = self.__class_weight
          weight0 = y*weight0[1] + ( 1-y)*weight0[0]

        if self.__focal_loss:
          # 簡単に分類成功しているものの重みを小さくする
          # prob = P(Y=1 | X) なので
          # y = 1の時   1-prob
          # y = 0の時    prob
          weight0 = (y)*(1-prob) + (1-y)*(prob)
          weight0 = (1 - weight0) ** self.__focal_loss
 
        if  self.__imbalanced_weight and sample_weight is not None:
            weight0 = sample_weight
            # ib_bunshi = abs(prob - y)
            # ib_bunbo = ((prob*(1-prob))).sum() + self.__lamda
            # ib = ib_bunshi /(ib_bunbo + 1e-5)
            # weight0 =  1/(ib+1e-5)
            # weight0 = len(weight0)*weight0/sum(weight0)

            if len(weight0) != len(y):
                raise ValueError(f"calc_loss でのサイズエラー{weight0.shape} {self.__y.shape}")


        g = ((y - prob)*weight0).sum()
        h = ((prob*(1-prob))*weight0).sum() + self.__lamda
        tmp = g**2
        return -0.5 * (tmp/h)
            
            
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
        # 引数で指定された特徴量としきい値を使って、そのしきい値以上のサンプルのy　と　しきい値未満のサンプルy
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
        time_start = time.time()
        kouho = self.feature_candidate(features)
        
        for feat_idx in kouho:
            now_feature = self.__X[:, feat_idx]
            # gc.collect()
            max_value,min_value = max(now_feature),min(now_feature)
            if max_value - min_value < abs(0.001):
                continue
            z = (now_feature - min_value )/(max_value - min_value) #0-1正規化
            #f = np.linspace(np.min(now_feature),np.max(now_feature),10)
            f = np.linspace(0.01, 0.99, 10)
            
            
            for value in f:
                information_gain = self.calc_information_gain(feat_idx, value)
                if gain_max < information_gain:  # 最大情報利得の更新
                    gain_max = information_gain
                    best_thrs = value * (max_value - min_value) + min_value # しきい値
                    best_f = feat_idx #　特徴量の列番号

          
        # del now_feature
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
                    imbalanced_weight=self.__imbalanced_weight,focal_loss=self.__focal_loss,
                    sample_weight = self.__sample_weight[idx_left]
                    )
                self.__right = Node_XgBoost(self.__X[idx_right], self.__y[idx_right], self.__log_odds[idx_right], self.__lamda,
                    self.__max_depth, position="right", class_weight=self.__class_weight,feature_select = self.__feature_select,
                    imbalanced_weight=self.__imbalanced_weight,focal_loss=self.__focal_loss,
                    sample_weight = self.__sample_weight[idx_right]
                    )

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

    def calc_hessian_value(self, data):  
        if self.__left is None and self.__right is None:
            return self.__total_hessian_with_lambda
        else:
            if data[self.__feature] >= self.__threshold:   
                return self.__left.calc_hessian_value(data)
            else:
                return self.__right.calc_hessian_value(data)

    
                
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

class OriginalXgBoostDecisoinTree:
    def __init__(self, log_odds, max_depth,class_weight=None, min_sample=1,
    feature_select='all',lamda=1, imbalanced_weight=False, focal_loss=False,
    sample_weight=None):
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
        self.sample_weight = sample_weight
        
        
    def fit(self, data, target):
        self.type = 'clasiffication' if len(np.unique(target)) < 3 else 'regression'
        initial_depth = 1
            
        if self.sample_weight is not None:
            self.tree = Node_XgBoost(
                data, target,log_odds = self.log_odds, lamda = self.lamda, max_depth = self.max_depth, 
                class_weight=self.class_weight,min_sample=self.min_sample,feature_select = self.feature_select,
                imbalanced_weight=self.imbalanced_weight, focal_loss=self.focal_loss,
                sample_weight = self.sample_weight
            )
        else:
            self.tree = Node_XgBoost(
                data, target,log_odds = self.log_odds, lamda = self.lamda, max_depth = self.max_depth, 
                class_weight=self.class_weight,min_sample=self.min_sample,feature_select = self.feature_select,
                imbalanced_weight=self.imbalanced_weight, focal_loss=self.focal_loss
            )
        self.tree.split(initial_depth)


    def predict(self, data):
        # for s in data:
        #     pred.append(self.tree.predict(s)) # tree.predictは木の末端のノードを返す

        pred =[ self.tree.predict(s) for s in data ]
        return np.array(pred)
    
    def calc_hessian_value(self, data):
        pred = [self.tree.calc_hessian_value(s) for s in data]
        # for s in data:
        #     pred.append(self.tree.calc_hessian_value(s)) # tree.predictは木の末端のノードを返す
        return np.array(pred)

    def calc_sample_weights(self, X):
        return 0




class InfluenceBalancedXgBoost:
    def __init__(self, max_depth,class_weight=None, min_sample=10, feature_select='all',lamda=1,
                n_estimator = 10,learning_rate = 1, 
                imbalanced_weight=False, switch_turn=0.5,focal_loss=False,sub_sample=0.25,metrics='accuracy'):
        self.max_depth = max_depth
        self.tree = None
        self.class_weight= class_weight
        self.min_sample=min_sample
        self.feature_select = feature_select
        self.lamda = lamda
        self.n_estimator = n_estimator
        self.learning_rate = learning_rate
        self.models = []
        
        self.imbalanced_weight = imbalanced_weight
        self.swicth_turn = switch_turn
        self.focal_loss = focal_loss
        self.sub_sample = sub_sample
        self.metrics = metrics


    def calc_current_hessian(self, f0, X, y, model):
      hessian_last = model.calc_hessian_value(X) # 過去木のヘッシアン
      hessian_str = [str(round(i,3)) for i in hessian_last] 

      prob = self.sigmoid(f0)
      prob_prob = prob *(1-prob)

      total = {}
      for i, j in zip(hessian_str, prob_prob):
          if i in total:
              total[i] += j
          else:
              total[i] = j

      hessians = np.array([ (total[i]+self.lamda) for i in hessian_str])
      return hessians
    
    def check_ealry_stopping(self, ealry_stopping, types='train'):
      """
        早期終了するべきならば、True まだ続けるべきならばFalseを返す
      """
      if types not in ['train', 'valid']:
        raise ValueError('types引数は train か validにしてください')
      if types == 'train':
        accuracys = self.train_accuracy_log
      else:
        accuracys = self.valid_accuracy_log
        
      if len(self.models)!= len(accuracys):
        ValueError("モデルと accuracysの数が一致しません")
      if ealry_stopping < len(accuracys):
        latest_log = accuracys[-(ealry_stopping+1):]
        first_value = latest_log[0]
        max_value = max(latest_log)
        if abs(first_value - max_value)< 1e-5:
          result2 = accuracys[:-(ealry_stopping)]
          if types == 'train':
             self.train_accuracy_log = result2
             self.train_accuracy_log = result2
          else:
             self.valid_accuracy_log = result2
          self.models = self.models[:-ealry_stopping]
          return True
        else:
          return False
      else:
        return False
       
    def fit(self,X, y, valid_data=None,ealry_stopping=5):
        prob = y.mean()
        self.train_accuracy_log = []
        self.valid_accuracy_log = []
        is_not_switch = True
        normal_train_cnt=0
        is_early_stopping = False # early_stoppingが動作しなかったらFalse        
        max_accuracy = 0
        X_valid,y_valid = valid_data

        if self.imbalanced_weight:
          swicth = int(self.n_estimator*self.swicth_turn)
          f0 =  np.ones(len(y)) * np.log( 1e-10 + (prob/(1-prob+ 1e-10)) )
        else:
          swicth = self.n_estimator
          f0 = np.zeros(len(y)) # np.ones(len(y)) * np.log(prob/(1-prob+ 1e-5))
        self.initial_log_odds = f0[0]


        for i in range(swicth):
          
          model = OriginalXgBoostDecisoinTree(
                log_odds=f0, max_depth=self.max_depth,lamda=self.lamda,
                min_sample=self.min_sample, feature_select=self.feature_select,class_weight=self.class_weight,
                imbalanced_weight=False, focal_loss=self.focal_loss
          )
          model.fit(X, y)
          self.models.append(model)
          f0 += self.learning_rate * model.predict(X)

          self.train_accuracy_log.append(self.calc_accuracy(X, y))
          test_accuracy = self.calc_accuracy(X_valid, y_valid)
          self.valid_accuracy_log.append(test_accuracy)

          if self.imbalanced_weight:
            """ inf-xgboostの場合、訓練データの精度改善を調べて、改善しなくなったら、フェイズ２に移行
            """
            if self.check_ealry_stopping(ealry_stopping, types='train'):                  
              self.best_itr = len(self.models)
              self.valid_accuracy_log = self.valid_accuracy_log[:len(self.models)]
              
              f0 = self.predict_log_odds(X)
              self.normal_train_cnt = len(self.models)
              is_early_stopping = True
              # print(f'早期終了します. ラウンド{i}')
              break
          else:
            """
              それ以外の場合、valid_dataが設定されていたら、valid_dataが改善しなくなったら、そこで打ち切り
            """
            if self.check_ealry_stopping(ealry_stopping, types='valid'):
              is_early_stopping = True
              # print(f'早期終了します. ラウンド{i}')
              break
          # print(f'フェーズ１ ラウンド{i} end')
          #### next loop #####
        else:
          # 早期終了が発動しなかった
          self.normal_train_cnt = len(self.models)
          is_early_stopping = False

        if self.imbalanced_weight:
            
            weighted_train_num = int(np.ceil(self.normal_train_cnt * (1-self.swicth_turn)/self.swicth_turn))
            phase1_max_idx = np.argmax(self.valid_accuracy_log)
            phase1_max_valid_score = np.max(self.valid_accuracy_log)

            # print(f'validation = {self.valid_accuracy_log}')
            # print(f'len(train_log) = {len(self.train_accuracy_log)} len(valid_log) = {len(self.valid_accuracy_log)}')
            # print(f'len(models) = {len(self.models)}')
            

            phase2_valid_scores = []
            # print(f'{weighted_train_num}回の第２段階目学習を行う')
            is_flag = True
            for j in range(weighted_train_num):
              # print(f'フェーズ２: j = {j}')
              model = OriginalXgBoostDecisoinTree(log_odds=f0, max_depth=self.max_depth,lamda=self.lamda,
                  min_sample=self.min_sample, feature_select=self.feature_select, class_weight=self.class_weight,
                  imbalanced_weight=True, focal_loss=self.focal_loss, sample_weight=np.ones(len(y))
              )
              model.fit(X, y)
              self.models.append(model)
              f0 += self.learning_rate * model.predict(X) #対数オッズ
              
              test_accuracy = self.calc_accuracy(X_valid, y_valid)
              phase2_valid_scores.append(test_accuracy)
              self.valid_accuracy_log.append(test_accuracy)

              
              if (j <= (ealry_stopping - 1)) and is_flag:
                if max(self.valid_accuracy_log) > phase1_max_valid_score:
                   is_flag = False
                  #  print(f'フェーズ２を{ealry_stopping}回,まわす中でフェーズ１の記録を更新した')
                  #  print(f'次の条件式 = {((j+1) <= (ealry_stopping - 1)) and is_flag}')
                else:
                   # print(f'フェーズ２を{ealry_stopping}回,まわす中でフェーズ１の記録を更新しnなかった')
                   self.models = self.models[:(phase1_max_idx + 1)]
                   self.train_accuracy_log = self.train_accuracy_log[:(phase1_max_idx + 1)]
                   self.valid_accuracy_log = self.valid_accuracy_log[:(phase1_max_idx + 1)]
                   break
              else:
                # print(f'フェーズ２第一チェックを突破したので第２チェックに入る')
                if self.check_ealry_stopping(ealry_stopping, types='valid'):
                  is_early_stopping = True
                  # print(f'フェーズ２：早期終了します. ラウンド{j}')
                  break
            else:
               is_early_stopping = False
               # print('第二フェーズでの早期終了は発動しなかった')
    def predict_log_odds(self,X):
        n = X.shape[0]        
        result = self.initial_log_odds
        result += np.array( [ self.learning_rate * m.predict(X) for m in self.models] ).sum(axis=0)
        return result
    
    def sigmoid(self, x):
        x[x < -700] = -700
        prob = 1/(1+np.exp(-x))
        return prob
    
    def predict_proba(self, X):
        log_odds = self.predict_log_odds(X)
        return self.sigmoid(log_odds)
      
      

    def predict(self,X):
        answer = self.predict_proba(X)
        return np.round(answer)

    def calc_accuracy(self,X,y):
        pred = self.predict(X)
        
        return np.mean(pred==y)
           
    
    def calc_f1_score(self, X,y):
        from sklearn.metrics import f1_score
        pred = self.predict(X)
        return f1_score(y_true=y,y_pred=pred)

