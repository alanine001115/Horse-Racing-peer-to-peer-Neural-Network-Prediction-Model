import numpy as np
import itertools
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, roc_auc_score,average_precision_score, log_loss
from sklearn.utils import resample
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, Flatten, Concatenate, LSTM, BatchNormalization, SpatialDropout1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import Huber
import os
import random
from ProfitStrategy import riskless_profit, profits_summary
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt 
import xgboost as xgb
from scipy import stats
import matplotlib as mpl
import shap

folder = "exported_csvs"

data = {
    os.path.splitext(f)[0]: pd.read_csv(os.path.join(folder, f))
    for f in os.listdir(folder) if f.endswith(".csv")
}

train_idx, val_idx, test_idx = np.array(data['train_indices']).ravel(), np.array(data['val_indices']).ravel() ,np.array(data['test_indices']).ravel()


def set_global_determinism(seed=5):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

def log_loss_ci(y_true, y_test_probs, n_bootstraps=1000, ci=95):
    set_global_determinism()
    scores = [log_loss(*resample(y_true, y_test_probs)) for _ in range(n_bootstraps)]
    alpha = (100 - ci) / 2
    return np.mean(scores), np.percentile(scores, [alpha, 100 - alpha])

def evaluate(y_test, y_pred, y_test_probs):
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    loss, (lower, upper) = log_loss_ci(y_test, y_test_probs)
    
    metrics = {
        'f1_score': f1,
        'precision_score' : precision,
        'recall_score': recall,
        'accuracy_score': accuracy,
        'log_loss': loss,
        'log_loss_ci': (lower, upper)
    }
    return metrics

def ROC_AUC(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    print("AUC-ROC:", auc_score)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0,1], [0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

    return 

def compute_saliency_map(model, X, sample_idx):
    X_sample = tf.convert_to_tensor(X[sample_idx:sample_idx+1], dtype=tf.float32)  # (1, T, F)

    with tf.GradientTape() as tape:
        tape.watch(X_sample)
        y_pred = model(X_sample, training=False)  

    grads = tape.gradient(y_pred, X_sample)  
    saliency = tf.abs(grads)[0].numpy()
    return saliency

def compute_aggregate_saliency(model, X, num_samples=100):
    N = X.shape[0]
    sample_indices = np.random.choice(N, size=min(num_samples, N), replace=False)

    saliency_list = []
    for idx in sample_indices:
        saliency = compute_saliency_map(model, X, idx)
        saliency_list.append(saliency)

    agg_saliency = np.mean(saliency_list, axis=0)
    return agg_saliency

def CNN_LSTM_Explain(model, X_test, file_name, feature_names, num_samples=200):

    agg_saliency = compute_aggregate_saliency(model, X_test, num_samples)
    
    plt.figure(figsize=(16, 9))
    plt.imshow(agg_saliency.T, cmap="plasma", aspect="auto")
    plt.colorbar(label="Average Saliency")
    plt.xlabel("Time step")
    plt.ylabel("Features")
    y_pos = np.arange(len(feature_names))
    plt.yticks(y_pos, feature_names)
    plt.title("Aggregated Saliency Map (avg over 200 samples)")
    plt.savefig(file_name, format="pdf", bbox_inches="tight")
    
    plt.show()

def feature_importance(model, X_test):
    xgb.plot_importance(model, importance_type='gain')
    plt.title("XGBoost Feature Importance (Gain)")
    plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    shap.summary_plot(shap_values, X_test, plot_type="bar")

    shap.summary_plot(shap_values, X_test)

    return {'explainer': explainer, 'shap_values': shap_values}

def feature_importance_table(model, X_test, feature_names, feature_size=5):
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
        
    importance = []
    for i in range(0, X_test.shape[1], feature_size):
        feature_shap = np.mean(np.abs(shap_values[:, i:i+feature_size]), axis=1)
        importance.append(np.mean(feature_shap))

    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(feature_names))
    plt.barh(y_pos, importance, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Mean SHAP value')
    plt.title(f'Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    importance_df = pd.DataFrame({
        'feature_names': feature_names,
        'feature_importance (%)': importance/np.sum(importance)*100
    })

    latex_code = importance_df.to_latex(
        index=False,
        position='h!',
        escape=False,
        float_format="%.1f"
    )
    
    return latex_code

def analyse_profit(returns_array):
    returns = np.array(returns_array)
    
    actual_bets = returns[returns != 0]
    
    if len(actual_bets) == 0:
        return {
            'total_return': 0,
            'average_return': 0,
            'confidence_interval': (0, 0),
            'n_bets': 0,
            'message': 'No actual bets placed (all zeros)'
        }
        
    total_return = np.sum(actual_bets)
    average_return = np.mean(actual_bets)
    n_bets = len(actual_bets)

    if n_bets > 1:
        std_err = stats.sem(actual_bets)
        ci = stats.t.interval(0.95, df=n_bets-1, loc=average_return, scale=std_err)
    else:
        ci = (average_return, average_return)
    
    return {'total_return': total_return,
            'average_return_per_bet': average_return,
            'positive_rate': np.sum(actual_bets>0) / n_bets,
            'confidence_interval_95': ci,
            'number_of_bets': n_bets,
            'total_observations': len(returns),
            'bets_ratio': n_bets / len(returns) }

def Prepare_Data(selected_features, target_feature, dict_data = data, 
                 train_idx = train_idx, val_idx = val_idx, test_idx = test_idx, window_size=5):
    
    X = np.stack([dict_data[feat].values for feat in selected_features], axis=-1)
    y = np.array(dict_data[target_feature].values)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    
    scaler_X = StandardScaler()
    
    for i in range(X_train.shape[-1]):
        X_train[:,:,i] = scaler_X.fit_transform(X_train[:,:,i])
        X_val[:,:,i] = scaler_X.transform(X_val[:,:,i])
        X_test[:,:,i] = scaler_X.transform(X_test[:,:,i])

    return X_train, X_val, X_test, y_train.ravel(), y_val.ravel(), y_test.ravel()

def Prepare_Data_XGBoost(selected_features, target_feature, baseline = 0, dict_data = data, 
                 train_idx = train_idx, val_idx = val_idx, test_idx = test_idx, window_size=5):
    
    X = np.stack([dict_data[feat].values for feat in selected_features], axis=-1)
    y = np.array(dict_data[target_feature].values)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    def static_feature(X):
        X_seq = []
        for i in range(X.shape[0]):  
            window = X[i, :, :]
            min_v = np.min(window, axis=0)
            max_v = np.max(window, axis=0)
            mean_v = np.mean(window, axis=0)
            last_v = X[i, -1, :]
            slope_v, _ =  np.polyfit([1,2,3,4,5], window, 1)

            stats_vector = np.concatenate([min_v, max_v, mean_v, last_v, slope_v])
            
            X_seq.append(stats_vector)
                
        return np.array(X_seq)

    if baseline == 1:
        def static_feature(X):
            X_seq = []
            for i in range(X.shape[0]):  
                window = X[i, :, :]
                last_v = X[i, -1, :]
                slope_v, _ =  np.polyfit([1,2,3,4,5], window, 1)
                mean_v = np.mean(window, axis=0)
                stats_vector = np.concatenate([last_v, slope_v, mean_v])
                X_seq.append(stats_vector)            
            return np.array(X_seq)

    return static_feature(X_train), static_feature(X_val), static_feature(X_test), y_train.ravel(), y_val.ravel(), y_test.ravel()

def best_threshold(y_val_true, y_val_probs, r = 0.01):
    thresholds = np.arange(0, 1.01, 0.01)
    scores = []

    for t in thresholds:
        y_val_pred = (y_val_probs >= t).astype(int)
        recall = recall_score(y_val_true, y_val_pred)
        precision = precision_score(y_val_true, y_val_pred) if (y_val_pred.sum() > 0) else 0

        if recall >= r:
            score = precision
        else:
            score = -np.inf

        scores.append(score)

    best_t = thresholds[np.argmax(scores)]
    print('val_set best precision:', max(scores))
    
    return best_t

def CNN_LSTM_Model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        
        Conv1D(filters=64, kernel_size=3, 
               padding='same', 
               activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        SpatialDropout1D(0.2),
        
        Conv1D(filters=32, kernel_size=3, 
               padding='same', 
               activation='relu'),
        BatchNormalization(),
        SpatialDropout1D(0.2),
        
        LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
    
        Dense(50, activation='relu'),
        Dropout(0.3),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss= 'binary_crossentropy'
    )
    
    return model

def CNN_LSTM_Results(selected_features, target_feature, dict_data = data, 
              train_idx = train_idx, test_idx = test_idx,
              epochs=100, batch_size=32, verbose=0):
    
    set_global_determinism()
    
    model = CNN_LSTM_Model( input_shape = (5, len(selected_features) ) )

    X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = \
    Prepare_Data(selected_features, target_feature)

    callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train.ravel())

    weight_dict = {0: class_weights[0], 1: class_weights[1]}

    model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
              class_weight=weight_dict,
              #validation_split=0.2,
              validation_data=(X_val_scaled, y_val),
              callbacks=callbacks,verbose=verbose)

    y_train_probs = model.predict(X_train_scaled).ravel()
    y_val_probs = model.predict(X_val_scaled).ravel()
    y_test_probs = model.predict(X_test_scaled).ravel()

    t = best_threshold( y_val , y_val_probs)

    y_pred = (y_test_probs >= t).astype(int)
    
    return {'forecast':y_pred, 
            'model':model, 
            'train_true':y_train.ravel(),
            'val_true':y_val,
            'test_true':y_test,
            'threshold':t,
            'metrics': evaluate(y_test, y_pred, y_test_probs),
            'train_probs':y_train_probs,
            'val_probs':y_val_probs,
            'test_probs':y_test_probs,
            'X_test': X_test_scaled}

def XGBoost_Results(selected_features, target_feature, baseline = 0, dict_data = data, 
              train_idx = train_idx, test_idx = test_idx, verbose = 0):

    set_global_determinism()

    X_train, X_val, X_test, y_train, y_val, y_test = Prepare_Data_XGBoost(selected_features,target_feature, baseline)

        
    best_score = -np.inf
    best_params = None
    best_model = None

    max_depths = [3, 4, 5, 6, 7]
    learning_rates = [0.01, 0.05, 0.1]

    for md, lr in itertools.product(max_depths, learning_rates):
        model = xgb.XGBClassifier(
            max_depth=md,
            learning_rate=lr,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='aucpr',
            scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
            early_stopping_rounds=20
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=verbose
        )
        
        # Evaluate on validation set
        val_probs = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, val_probs)
        
        if score > best_score:
            best_score = score
            best_params = {'max_depth': md, 'learning_rate': lr}
            best_model = model

    print(f"Best params: {best_params}, Best PR-AUC: {best_score:.4f}")
    
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_val_probs = model.predict_proba(X_val)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]
    
    t = best_threshold(y_val, y_val_probs)
    
    y_pred = (y_test_probs >= t).astype(int)
    
    return {'forecast':y_pred, 
            'model':model, 
            'train_true':y_train.ravel(),
            'val_true':y_val,
            'test_true':y_test,
            'threshold':t,
            'metrics': evaluate(y_test, y_pred, y_test_probs),
            'train_probs':y_train_probs,
            'val_probs':y_val_probs,
            'test_probs':y_test_probs,
            'X_test': X_test}

def Meta_Model(model_1, model_2, max_depth=2, verbose=0):
    X_train = np.column_stack([model_1['train_probs'], model_2['train_probs']])
    X_val = np.column_stack([model_1['val_probs'], model_2['val_probs']])
    X_test = np.column_stack([model_1['test_probs'], model_2['test_probs']])
    y_train, y_val, y_test = model_1['train_true'], model_1['val_true'], model_1['test_true']
    
    model = xgb.XGBClassifier(
        max_depth=max_depth,
        learning_rate=0.05,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',  
        early_stopping_rounds=20
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )
    
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_val_probs = model.predict_proba(X_val)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]
    
    t = best_threshold(y_val, y_val_probs)
    
    y_pred = (y_test_probs >= t).astype(int)
    
    return {'forecast':y_pred, 
            'model':model, 
            'train_true':y_train.ravel(),
            'val_true':y_val,
            'test_true':y_test,
            'threshold':t,
            'metrics': evaluate(y_test, y_pred, y_test_probs),
            'train_probs':y_train_probs,
            'val_probs':y_val_probs,
            'test_probs':y_test_probs}