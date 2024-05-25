import numpy as np
import xgboost as xgb
import glob
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
class InfluenceBalancedXGB:
    def __init__(self, n_estimator=100, switch_turn=50, metrics="f1"):
        self.n_estimator=n_estimator
        self.max_switch_turn = switch_turn
        if metrics in ["f1", "precision", "g-mean"]:
            self.metrics = metrics
        else:
            raise ValueError("metricsの指定が誤り")
        
    def calc_weighted_accuracy(self, dtest, bst):
        y = dtest.get_label()
        y_pred = np.round(bst.predict(dtest))
        score = np.mean(precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[1])
        return score

    def calc_recall(self, dtest, bst):
        y = dtest.get_label()
        y_pred = np.round(bst.predict(dtest))
        score = precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[1][1]
        return score

    def score(self, dtest):
        y = dtest.get_label()
        y_pred = np.round(self.bst.predict(dtest))
        if self.metrics == "precision":
            score = np.mean(precision_recall_fscore_support(y_pred=y_pred, y_true=y,zero_division=0)[0])
        elif self.metrics == "g-mean":
            t = precision_recall_fscore_support(y_pred=y_pred, y_true=y, zero_division=0)[1]
            score = np.sqrt(t[0]*t[1])
        else:
            score = (precision_recall_fscore_support(y_pred=y_pred, y_true=y, zero_division=0)[2][1])
        # print(score)
        return score
    def calc_weight(self, bst, dtrain):
        y = dtrain.get_label()
        lamda = self.params['reg_lambda']

        prob = bst.predict(dtrain)
        leaf_indices = bst.predict(dtrain, pred_leaf=True)
        if len(leaf_indices.shape) > 1:
            leaf_indices = leaf_indices[:, -1]
        results = {}
        for p, idx in zip(prob, leaf_indices):
            if idx in results:
                results[int(idx)] += p*(1-p)
            else:
                results[int(idx)] = p*(1-p)
        hessian = pd.Series(leaf_indices).map(results).values
        ib = abs(prob - y) / (hessian + lamda + 1e-5)
        weight = (hessian + lamda) / (abs(prob - y)+1e-5)
        return weight, ib
         
    def fit(self, dtrain, dtest, params,ealry_stoping=5):
        self.models=[]
        params["eta"]=1
        self.params = params
        X_train = dtrain.get_data().toarray()
        y_train = dtrain.get_label()
        self.train_ib = []
        self.train_score = []
        self.test_score = []
        self.test_ib = []
        # 初期モデルをトレーニング
        bst = None
        for i in range(1, self.n_estimator+1):
            # 以前のモデルから学習を再開
            if bst is None:
                bst = xgb.train(params, dtrain, num_boost_round=1)
            else:
                bst = xgb.train(
                    params, dtrain, num_boost_round=1,     
                    xgb_model=bst
                    )
            self.models.append(bst)
            self.bst = bst
            weights, ib = self.calc_weight(bst, dtrain)
            self.train_ib.append(ib)
            th = (1 - y_train.mean())
            train_recall = self.calc_recall(dtrain, bst)
            # self.calc_weighted_accuracy(dtrain, bst)
            self.train_score.append(self.score(dtrain))
            self.test_score.append(self.score(dtest))

            if th <= train_recall:
                # print("train_recall",th, train_recall)
                self.phase1_end = i
                break
        else:
            self.phase1_end = self.n_estimator
        
        current_score = self.score(dtest=dtest)
        # self.calc_weighted_accuracy(dtest=dtest, bst=bst)
        count_no_min = 0
        switch_turn = self.max_switch_turn
        self.temp_score= []
        for i in range(self.max_switch_turn):
            # ここで新しい重みを計算 (例としてランダムに重みを変更)
            weights, ib = self.calc_weight(bst, dtrain)
            # 新しいDMatrixを作成
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
            # 以前のモデルから学習を再開
            bst = xgb.train(params, dtrain, num_boost_round=1, 
                            xgb_model=bst,verbose_eval=False)
            test_score = self.score(dtest=dtest)
            self.temp_score.append(test_score)
            

            if test_score > current_score:
                current_sco = test_score
                count_no_min = 0
            else:
                count_no_min += 1
            if count_no_min >= ealry_stoping:
                switch_turn = i-(ealry_stoping-1)
                break
        self.bst = bst
        for j in range(switch_turn):
            # ここで新しい重みを計算 (例としてランダムに重みを変更)
            weights, ib = self.calc_weight(bst0, dtrain)
            self.test_ib.append(ib)
            # 新しいDMatrixを作成
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
            # 以前のモデルから学習を再開
            bst0 = xgb.train(params, dtrain, num_boost_round=1, 
                            xgb_model=bst0,verbose_eval=False)
            
            test_score = self.score(dtest=dtest)
            train_score = self.score(dtest=dtrain)
            self.test_score.append(test_score)
            self.train_score.append(test_score)
            

            self.models.append(bst0)
        self.phase2_end = switch_turn
        
    def predict(self, dtest):
         return self.bst.predict(dtest)