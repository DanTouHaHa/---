# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, ParameterGrid, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import optuna
from rdkit import Chem
from rdkit.Chem import AllChem


# 定义评估模型性能的函数
def get_error_evaluation(model, x_test, y_test):
    predictions = model.predict(x_test)
    absolute_errors = np.abs(y_test - predictions)
    squared_errors = (y_test - predictions) ** 2
    mae_mean = np.mean(absolute_errors) / len(predictions)
    mse_mean = np.mean(squared_errors) / len(predictions)
    mae_std = np.std(absolute_errors) / len(predictions)
    mse_std = np.std(squared_errors) / len(predictions)
    return mae_mean, mse_mean, mae_std, mse_std

# 定义优化模型参数的函数
def optimize_model(x_train, y_train, Model, model_params, n_trials=30):
    import warnings; warnings.filterwarnings("ignore")
    def model_objective(trial):
        trial_model_params = {name: trial.suggest_categorical(name, list(choice)) for name, choice in model_params.items()}
        model = Model(**trial_model_params)
        predictions = cross_val_predict(model, x_train, y_train, cv=5)
        mse = mean_squared_error(y_train, predictions)
        return mse / len(predictions)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(model_objective, n_trials=n_trials)  # 可调整试验次数
    print(Model.__name__, "finishes super-parameter optimization!\n", "*"*60)
    return Model(**study.best_params)

# 根据药物simles生成分子指纹特征
def get_smiles_feature(data):
    smi_dicts={ 
    '奥氮平': "CN1CCN(CC1)C1=NC2=CC=CC=C2NC2=C1C=C(C)S2",
    '氟哌啶醇': "OC1(CCN(CCCC(=O)C2=CC=C(F)C=C2)CC1)C1=CC=C(Cl)C=C1", 
    '齐拉西酮': "ClC1=C(CCN2CCN(CC2)C2=NSC3=CC=CC=C23)C=C2CC(=O)NC2=C1", 
    '阿立哌唑': "ClC1=CC=CC(N2CCN(CCCCOC3=CC4=C(CCC(=O)N4)C=C3)CC2)=C1Cl", 
    '利培酮': "CC1=C(CCN2CCC(CC2)C2=NOC3=C2C=CC(F)=C3)C(=O)N2CCCCC2=N1", 
    '喹硫平': "OCCOCCN1CCN(CC1)C1=NC2=CC=CC=C2SC2=CC=CC=C12", 
    '奋乃静': "OCCN1CCN(CCCN2C3=CC=CC=C3SC3=C2C=C(Cl)C=C3)CC1"
    }
    smi_finger={}
    for key in smi_dicts:
        smi_mol = Chem.MolFromSmiles(smi_dicts[key])
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(smi_mol, 2, nBits=16)
        smi_finger[key] = [int(bit) for bit in fingerprint.ToBitString()]

    return np.array(data.map(smi_finger).to_list())

# 载入数据
file_path = '回归预测.xlsx'
train_data = pd.read_excel(file_path, sheet_name=0)
test_data = pd.read_excel(file_path, sheet_name=1)

# 数据预处理
train_drug_encoded = get_smiles_feature(train_data.iloc[:, 30])
test_drug_encoded = get_smiles_feature(test_data.iloc[:, 30]) 

x_train = np.hstack((train_data.iloc[:, :30].values, train_drug_encoded))
y_train = train_data.iloc[:, 31].values
x_test = np.hstack((test_data.iloc[:, :30].values, test_drug_encoded))
y_test = test_data.iloc[:, 31].values

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# RF、GBM、MLP模型参数网格定义
rf_params = {'n_estimators': [50, 100, 200, 300, 1000], 'max_depth': [5, 10, 20, 30, 100]}
gbm_params = {'n_estimators': [50, 100, 150, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5], 'max_depth': [3, 5, 7]}
mlp_params = {'hidden_layer_sizes': [(50), (100), (100, 50)], 'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
              'alpha': [0.0001, 0.001, 0.01], 'learning_rate': ['constant', 'adaptive'], 'max_iter': [1000]}

# RF、GBM、MLP模型初始化和参数优化
best_rf_model = optimize_model(x_train, y_train, RandomForestRegressor, rf_params)
best_rf_model.fit(x_train, y_train)
best_gbm_model = optimize_model(x_train, y_train, GradientBoostingRegressor, gbm_params)
best_gbm_model.fit(x_train, y_train)
best_mlp_model = optimize_model(x_train, y_train, MLPRegressor, mlp_params)
best_mlp_model.fit(x_train, y_train)

# XGBoost模型优化
xgb_param_grid = {'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5], 'max_depth': [3, 5, 7]}
xgb_cv_results = {}
for params in ParameterGrid(xgb_param_grid):
    cv_results = xgb.cv(params, xgb.DMatrix(x_train, label=y_train), num_boost_round=50, nfold=5, metrics='rmse', early_stopping_rounds=10)
    xgb_cv_results[tuple(params.values())] = cv_results.iloc[-1, 0]
best_xgb_params = min(xgb_cv_results, key=xgb_cv_results.get)
best_xgb_model = xgb.XGBRegressor(learning_rate=best_xgb_params[0], max_depth=int(best_xgb_params[1]), n_estimators=50)
best_xgb_model.fit(x_train, y_train)

# 堆叠模型
stacking_params={"final_estimator": [Ridge(), LinearRegression(), SVR(), MLPRegressor(), GradientBoostingRegressor(), RandomForestRegressor(), DecisionTreeRegressor()],
                 "estimators": [[('rf', best_rf_model), ('gbm', best_gbm_model), ('xgb', best_xgb_model), ('mlp', best_mlp_model)]]}
best_stacking_model = optimize_model(x_train, y_train, StackingRegressor, stacking_params, n_trials=7)
best_stacking_model.fit(x_train, y_train)

# Training and Evaluating Models
rf_mae_mean, rf_mse_mean, rf_mae_std, rf_mse_std = get_error_evaluation(best_rf_model, x_test, y_test)
gbm_mae_mean, gbm_mse_mean, gbm_mae_std, gbm_mse_std = get_error_evaluation(best_gbm_model, x_test, y_test)
xgb_mae_mean, xgb_mse_mean, xgb_mae_std, xgb_mse_std = get_error_evaluation(best_xgb_model, x_test, y_test)
mlp_mae_mean, mlp_mse_mean, mlp_mae_std, mlp_mse_std = get_error_evaluation(best_mlp_model, x_test, y_test)
stack_mae_mean, stack_mse_mean, stack_mae_std, stack_mse_std = get_error_evaluation(best_stacking_model, x_test, y_test)

print(f"rf_mae_mean:{rf_mae_mean}, rf_mse_mean:{rf_mse_mean}, rf_mae_std:{rf_mae_std}, rf_mse_std:{rf_mse_std}")
print(f"gbm_mae_mean:{gbm_mae_mean}, gbm_mse_mean:{gbm_mse_mean}, gbm_mae_std:{gbm_mae_std}, gbm_mse_std:{gbm_mse_std}")
print(f"xgb_mae_mean:{xgb_mae_mean}, xgb_mse_mean:{xgb_mse_mean}, xgb_mae_std:{xgb_mae_std}, xgb_mse_std:{xgb_mse_std}")
print(f"mlp_mae_mean:{mlp_mae_mean}, mlp_mse_mean:{mlp_mse_mean}, mlp_mae_std:{mlp_mae_std}, mlp_mse_std:{mlp_mse_std}")
print(f"stacking_mae_mean:{stack_mae_mean}, stacking_mse_mean:{stack_mse_mean}, stacking_mae_std:{stack_mae_std}, stacking_mse_std:{stack_mse_std}")