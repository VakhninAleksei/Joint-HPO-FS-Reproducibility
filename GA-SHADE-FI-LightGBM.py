from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.kernel_approximation import RBFSampler
from typing import List, Tuple, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from hyppo.independence import Hsic
from lightgbm import LGBMRegressor
from dataclasses import dataclass
from sklearn.svm import LinearSVR
from openml.tasks import TaskType
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import cauchy
import lightgbm as lgb
import pandas as pd
import numpy as np
import datetime
import warnings
import openml
import random
import math
import time
import copy
import os
import re

ROOT_DIR = "openml_regression_dump"
os.makedirs(ROOT_DIR, exist_ok=True)

def _make_progress_print_callback(run_idx: int, every: int = 10):
    t0 = time.time()
    def callback(study: optuna.Study, trial: optuna.Trial):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if every and (n_complete % every == 0):
            dt = time.time() - t0
            best = study.best_value if study.best_trial is not None else None
            print(f"[run {run_idx+1:02d}] trials={n_complete} best={best:.6f} elapsed={dt:.1f}s")
    return callback

def save_task_by_tid(tid: int, root_dir: str = ROOT_DIR):
    task_dir = os.path.join(root_dir, str(tid))
    splits_dir = os.path.join(task_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    X_path = os.path.join(task_dir, "X.csv")
    y_path = os.path.join(task_dir, "y.csv")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print("  -> already exists, skipping")
        return

    task = openml.tasks.get_task(tid)

    if task.task_type_id != TaskType.SUPERVISED_REGRESSION:
        raise ValueError(f"Task {tid} is not a supervised regression task.")

    X, y = task.get_X_and_y(dataset_format="dataframe")

    n_repeats, n_folds, n_samples = task.get_split_dimensions()
    print(f"  splits: {n_repeats} repeats x {n_folds} folds; n_samples={n_samples}")

    if not isinstance(y, pd.DataFrame):
        y_df = y.to_frame(name="y")
    else:
        y_df = y

    X.to_csv(X_path, index=False)
    y_df.to_csv(y_path, index=False)

    for repeat in range(n_repeats):
        for fold in range(n_folds):
            train_idx, test_idx = task.get_train_test_split_indices(
                repeat=repeat, fold=fold, sample=0)

            split_folder = os.path.join(splits_dir, f"repeat_{repeat}_fold_{fold}")
            os.makedirs(split_folder, exist_ok=True)

            pd.DataFrame({"train_idx": train_idx}).to_csv(
                os.path.join(split_folder, "train_idx.csv"),
                index=False)
            pd.DataFrame({"test_idx": test_idx}).to_csv(
                os.path.join(split_folder, "test_idx.csv"),
                index=False)

    print(f"  -> saved to {task_dir}")

ROOT_DIR = "openml_regression_dump"

@dataclass(frozen=True)
class TaskCache:
    X: np.ndarray                       # shape (n, d)
    y: np.ndarray                       # shape (n,)
    splits: List[Tuple[np.ndarray, np.ndarray]]  # (train_idx, test_idx)

#LOAD ONE TIME

def load_saved_task_cache(
    tid: int,
    root_dir: str = ROOT_DIR,
    use_numpy: bool = True
) -> TaskCache:
    task_dir = os.path.join(root_dir, str(tid))
    X_path = os.path.join(task_dir, "X.csv")
    y_path = os.path.join(task_dir, "y.csv")
    splits_dir = os.path.join(task_dir, "splits")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.isdir(splits_dir)):
        raise FileNotFoundError(f"Missing files for tid={tid} in {task_dir}")

    # X, y
    X_df = pd.read_csv(X_path)
    y_df = pd.read_csv(y_path)

    # y -> 1d numpy
    if y_df.shape[1] == 1:
        y_series = y_df.iloc[:, 0]
    else:

        raise ValueError(f"Expected single-target y, got shape {y_df.shape}")

    if use_numpy:
        X = X_df.to_numpy()
        y = y_series.to_numpy()
    else:
        X = X_df
        y = y_series

    split_folders = []
    for name in os.listdir(splits_dir):
        m = re.match(r"repeat_(\d+)_fold_(\d+)$", name)
        if m:
            split_folders.append((int(m.group(1)), int(m.group(2)), name))
    if not split_folders:
        raise RuntimeError(f"No split folders found in {splits_dir}")

    split_folders.sort()

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for _, _, folder_name in split_folders:
        folder = os.path.join(splits_dir, folder_name)
        train_idx = pd.read_csv(os.path.join(folder, "train_idx.csv"))["train_idx"].to_numpy(dtype=np.int64)
        test_idx  = pd.read_csv(os.path.join(folder, "test_idx.csv"))["test_idx"].to_numpy(dtype=np.int64)
        splits.append((train_idx, test_idx))

    return TaskCache(X=X, y=y, splits=splits)
    
TID = 4902
save_task_by_tid(TID)
cache = load_saved_task_cache(TID)  # one time

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def fitness_lgbm_mae_all_features_HPO(cache, solution, n1, n2):
    solution = np.asarray(solution, dtype=float)

    if len(solution) < n1 + n2:
        raise ValueError(
            f"Solution length is too small: len(solution)={len(solution)}, expected at least {n1+n2}")

    X = np.asarray(cache.X)
    y = np.asarray(cache.y)

    # --- 1) HP ---
    hp = solution[:n1].copy()

    # --- 2) mask of features ---
    feature_mask = solution[n1:n1+n2]
    feature_mask = (feature_mask >= 0.5).astype(int)
    n2_ones = int(feature_mask.sum())

    if n2 != X.shape[1]:
        raise ValueError(
            f"n2 must match number of features: n2={n2}, X.shape[1]={X.shape[1]}"
        )

    if n2_ones == 0:
        return float("inf"), float("inf"), 0

    feature_idx = np.flatnonzero(feature_mask)

    mx = feature_idx.max()
    if mx >= X.shape[1]:
        raise ValueError(
            f"Feature index out of bounds: max(feature_idx)={mx}, X.shape[1]={X.shape[1]}")

    X = X[:, feature_idx]

    boosting_type = ['gbdt', 'dart', 'goss']
    bt_idx = int(hp[0])
    bt_idx = 0 if bt_idx <= 0 else (2 if bt_idx >= 2 else 1)
    bt = boosting_type[bt_idx]

    subsample_val = float(hp[7])
    if bt == 'goss':
        subsample_val = 1.0

    FIXED = dict(
        objective='regression_l1',
        boosting_type=bt,
        learning_rate=float(hp[1]),
        n_estimators=int(hp[2]),
        num_leaves=int(hp[3]),
        max_depth=int(hp[4]),
        min_child_samples=int(hp[5]),
        min_child_weight=float(hp[6]),
        subsample=subsample_val,
        subsample_freq=int(hp[8]),
        colsample_bytree=float(hp[9]),
        reg_alpha=float(hp[10]),
        reg_lambda=float(hp[11]),
        min_split_gain=float(hp[12]),
        max_bin=int(hp[13]),
        random_state=42,
        n_jobs=-1,
        verbosity=-1)

    scores = []

    callbacks = []
    if bt != 'dart':
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]

    for train_idx, valid_idx in cache.splits:
        X_tr, X_val = X[train_idx], X[valid_idx]
        y_tr, y_val = y[train_idx], y[valid_idx]

        model = lgb.LGBMRegressor(**FIXED)

        try:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='l1',
                callbacks=callbacks)
                
            y_pred = model.predict(X_val, validate_features=False)
            scores.append(mean_absolute_error(y_val, y_pred))
        except Exception:
            return float("inf"), float("inf"), n2_ones

    forecasting_error_1 = float(np.mean(scores))
    forecasting_error_2 = forecasting_error_1 * (1 + 0.005*(n2_ones))

    return float(forecasting_error_1), float(forecasting_error_2), n2_ones
    
TID = 4902
save_task_by_tid(TID)
cache = load_saved_task_cache(TID)

a = [
    0,        # boosting_type index: 0 (gbdt)
    0.005,    # learning_rate
    200,      # n_estimators
    31,       # num_leaves
    2,        # max_depth
    10,       # min_child_samples
    1e-3,     # min_child_weight
    0.4,      # subsample
    0,        # subsample_freq (0 = disable)
    0.4,      # colsample_bytree
    0.0,      # reg_alpha
    0.0,      # reg_lambda
    0.0,      # min_split_gain
    64        # max_bin
]

b = [
    2.99,      # boosting_type index: 2 (goss) 
    0.2,      # learning_rate
    1000,     # n_estimators
    255,      # num_leaves
    16,       # max_depth
    200,      # min_child_samples
    10.0,     # min_child_weight
    1.0,      # subsample
    5,        # subsample_freq
    1.0,      # colsample_bytree
    10.0,     # reg_alpha
    50.0,     # reg_lambda
    1.0,      # min_split_gain
    1024      # max_bin
]

n1 = 14
n2 = cache.X.shape[1]

def get_feature_generation_probabilities(
    TID,
    total_points=100,
    new_min=0.1,
    new_max=0.7,
    mi_random_state=42,
    plot=False
):
    def distance_correlation(x, y):
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        mask = np.isfinite(x[:, 0]) & np.isfinite(y[:, 0])
        x = x[mask]
        y = y[mask]

        n = x.shape[0]
        if n < 2:
            return np.nan

        a = np.abs(x - x.T)
        b = np.abs(y - y.T)

        A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

        dcov2 = (A * B).sum() / (n * n)
        dvarx2 = (A * A).sum() / (n * n)
        dvary2 = (B * B).sum() / (n * n)

        if dvarx2 <= 0 or dvary2 <= 0:
            return 0.0

        dcor = np.sqrt(dcov2) / np.sqrt(np.sqrt(dvarx2) * np.sqrt(dvary2))
        return float(np.clip(dcor, 0.0, 1.0))

    def proportional_ranks(score_series, total_points=100):
        score_series = score_series.copy()

        if np.all(~np.isfinite(score_series)) or score_series.fillna(0).sum() == 0:
            return pd.Series(0, index=score_series.index, dtype=int)

        score_series = score_series.fillna(0)
        sum_scores = score_series.sum()

        raw_ranks = total_points * score_series / sum_scores
        int_ranks = np.floor(raw_ranks).astype(int)
        remainder = raw_ranks - int_ranks

        missing = total_points - int_ranks.sum()
        if missing > 0:
            add_indices = remainder.nlargest(missing).index
            int_ranks.loc[add_indices] += 1

        return int_ranks.astype(int)

    def compute_hsic_series(X, y):
        tester = Hsic()
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)

        hsic_vals = []
        for j in range(X.shape[1]):
            x_arr = np.asarray(X.iloc[:, j].values, dtype=float).reshape(-1, 1)

            mask = np.isfinite(x_arr[:, 0]) & np.isfinite(y_arr[:, 0])
            x_clean = x_arr[mask]
            y_clean = y_arr[mask]

            if x_clean.shape[0] < 2:
                hsic_vals.append(np.nan)
                continue

            try:
                stat = tester.statistic(x_clean, y_clean)
            except Exception:
                stat = np.nan

            hsic_vals.append(stat)

        return pd.Series(hsic_vals, index=X.columns).clip(lower=0)

    def safe_abs_corr(func, x, y):
        try:
            val = func(x, y)[0]
            if pd.isna(val):
                return 0.0
            return abs(val)
        except Exception:
            return 0.0

    def filter_ranking_regression(X, y, total_points=100, mi_random_state=0):
        n_features = X.shape[1]
        scores = {}

        scores["pearson_abs"] = np.array([
            safe_abs_corr(pearsonr, X.iloc[:, j], y) for j in range(n_features)])

        scores["spearman_abs"] = np.array([
            safe_abs_corr(spearmanr, X.iloc[:, j], y) for j in range(n_features)])

        scores["kendall_abs"] = np.array([
            safe_abs_corr(kendalltau, X.iloc[:, j], y) for j in range(n_features)])

        try:
            mi = mutual_info_regression(X, np.asarray(y).ravel(), random_state=mi_random_state)
            mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            mi = np.zeros(n_features, dtype=float)
        scores["mi"] = mi

        y_arr = np.asarray(y).ravel()
        scores["dcor"] = np.array([distance_correlation(X.iloc[:, j].values, y_arr) for j in range(n_features)])
        scores["dcor"] = np.nan_to_num(scores["dcor"], nan=0.0, posinf=0.0, neginf=0.0)

        hsic_vals = compute_hsic_series(X, y).values
        hsic_vals = np.nan_to_num(hsic_vals, nan=0.0, posinf=0.0, neginf=0.0)
        scores["hsic"] = hsic_vals

        df = pd.DataFrame(scores, index=X.columns)

        df["pearson_final_rank"] = proportional_ranks(df["pearson_abs"], total_points=total_points)
        df["spearman_final_rank"] = proportional_ranks(df["spearman_abs"], total_points=total_points)
        df["kendall_final_rank"] = proportional_ranks(df["kendall_abs"], total_points=total_points)
        df["mi_final_rank"] = proportional_ranks(df["mi"], total_points=total_points)
        df["dcor_final_rank"] = proportional_ranks(df["dcor"], total_points=total_points)
        df["hsic_final_rank"] = proportional_ranks(df["hsic"], total_points=total_points)

        df["total_rank"] = df[
            [
                "pearson_final_rank",
                "spearman_final_rank",
                "kendall_final_rank",
                "mi_final_rank",
                "dcor_final_rank",
                "hsic_final_rank",
            ]
        ].sum(axis=1)

        return df.sort_values("total_rank", ascending=False)

    save_task_by_tid(TID)
    cache = load_saved_task_cache(TID)

    X = cache.X
    y = cache.y

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(np.asarray(X), columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
    else:
        X = X.copy()

    y = pd.Series(np.asarray(y).ravel(), name="target")

    feature_names = X.columns.tolist()
    n_features = X.shape[1]
    n_folds = len(cache.splits)

    fold_feature_scores = np.zeros((n_folds, n_features), dtype=float)

    for fold_id, (train_idx, valid_idx) in enumerate(cache.splits):
        X_tr = X.iloc[train_idx].copy()
        y_tr = y.iloc[train_idx].copy()

        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)

        fold_result = filter_ranking_regression(X_tr_scaled, y_tr.values, total_points=total_points, mi_random_state=mi_random_state + fold_id)
        fold_feature_scores[fold_id, :] = fold_result.loc[feature_names, "total_rank"].values

    feature_total_scores = fold_feature_scores.sum(axis=0)
    vals = pd.Series(feature_total_scores, index=feature_names)

    old_min = vals.min()
    old_max = vals.max()

    if old_max > old_min:
        normalized_prob = new_min + (vals - old_min) * (new_max - new_min) / (old_max - old_min)
    else:
        normalized_prob = pd.Series(new_min, index=vals.index, dtype=float)

    normalized_prob = normalized_prob.sort_values(ascending=False)

    if plot:
        sort_idx = np.argsort(-feature_total_scores)
        feature_names_sorted = [feature_names[i] for i in sort_idx]
        fold_feature_scores_sorted = fold_feature_scores[:, sort_idx]

        plt.figure(figsize=(max(12, n_features * 0.4), 7))
        x_pos = np.arange(n_features)
        bottom = np.zeros(n_features)
        colors = plt.cm.tab20(np.linspace(0, 1, n_folds))

        for fold_id in range(n_folds):
            plt.bar(
                x_pos,
                fold_feature_scores_sorted[fold_id],
                bottom=bottom,
                color=colors[fold_id],
                label=f"Fold {fold_id + 1}",
                width=0.8
            )
            bottom += fold_feature_scores_sorted[fold_id]

        totals = fold_feature_scores_sorted.sum(axis=0)
        for i, total in enumerate(totals):
            plt.text(i, total, f"{int(total)}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.xticks(x_pos, feature_names_sorted, rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Total rank per fold")
        plt.title("Stacked bar of feature ranking scores across CV folds")
        plt.legend(title="Folds")
        plt.tight_layout()
        plt.show()

        ax = normalized_prob.plot(kind="bar", figsize=(12, 6))
        for i, v in enumerate(normalized_prob):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.xlabel("Features")
        plt.ylabel("Probability for feature generation")
        plt.title("Normalized feature probabilities")
        plt.show()

    return normalized_prob
    

def init_population(population, pop_size, N, a, b, variable_type, prob, n1):
    prob = np.asarray(prob)
    half = int (pop_size * 0.7)
    bool_idx = [k for k in range(N) if variable_type[k] == 'bool']

    for i in range(pop_size):
        for j in range(N):
            if variable_type[j] == 'real':
                population[i][j] = random.random() * (b[j] - a[j]) + a[j]
            elif variable_type[j] == 'bool':
                if i < half:
                    #first part using prob
                    p = prob[j - n1]
                else:
                    #secind part using 0.5 prob
                    p = 0.5
                population[i][j] = 1 if random.random() < p else 0

        # fix infeasible solutions
        if sum(population[i][k] for k in bool_idx) == 0:
            if i < half:
                best = max(bool_idx, key=lambda k: prob[k - n1])
            else:
                best = random.choice(bool_idx)
            population[i][best] = 1
            
def generate_indices(pop_size, A, p, i):
    r1=int(random.random()*pop_size)
    r2=int(random.random()*(pop_size+A))
    max_best_number = int(p[i]*pop_size)
    bests_fitness = np.argsort(fitness)[:max_best_number]
    pbest = np.random.choice(bests_fitness)
    while (r1 == r2 or i == r1 or i == r2 ):
        r1=int(random.random()*pop_size)
        r2=int(random.random()*(pop_size+A))

    return r1, r2, pbest

def borders (v, population, N, a, b, variable_type):
    for i in range(0, pop_size):
        for j in range (0, N):
            if (variable_type[j] == 'real'):
                if (v[i][j] > b[j]):
                    v[i][j] = (b[j]+population[i][j])/2
                    v[i][j] = min(v[i][j], b[j])  
                if (v[i][j] < a[j]):
                    v[i][j] = (a[j]+population[i][j])/2
                    v[i][j] = max(v[i][j], a[j])  
            if (variable_type[j] == 'int'):
                if (v[i][j] > b[j]):
                    v[i][j] = int((b[j]+population[i][j])/2)
                    v[i][j] = min(v[i][j], b[j])  
                if (v[i][j] < a[j]):
                    v[i][j] = int((a[j]+population[i][j])/2)
                    v[i][j] = max(v[i][j], a[j])  
def isNaN(num):
    return num != num
    
for TID in [4926, 4937, 4939, 4941, 360064, 360066, 360069, 360080, 360083, 362368,363215, 363746, 363752]:#=  
    save_task_by_tid(TID)
    cache = load_saved_task_cache(TID)
    
    n1 = 14
    n2 = cache.X.shape[1]
        
    N = n1 + n2
    pop_size = 50
    
    hyperparams = []
    features = []
    
    for i in range(0,n1):
        hyperparams.append('r'+str(i))
    for i in range(0,n2):
        features.append('f'+str(i))
    
    solution_mask = hyperparams + features
    
    
    
    RUNS = 15
    MAX_RUNS = 15
    columns = ['evaluation', 'MAE_val', 'Penalty-based_fitness', 'features_n']+solution_mask
    
    
    variable_type =  ['real']*n1+['bool']*n2
    a = [
        0,        # boosting_type index: 0 (gbdt)
        0.005,    # learning_rate
        200,      # n_estimators
        31,       # num_leaves
        2,        # max_depth
        10,       # min_child_samples
        1e-3,     # min_child_weight
        0.4,      # subsample
        0,        # subsample_freq (0 = disable)
        0.4,      # colsample_bytree
        0.0,      # reg_alpha
        0.0,      # reg_lambda
        0.0,      # min_split_gain
        64        # max_bin
    ]
    
    b = [
        2.99,        # boosting_type index: 2 (goss)
        0.2,      # learning_rate
        1000,     # n_estimators
        255,      # num_leaves
        16,       # max_depth
        200,      # min_child_samples
        10.0,     # min_child_weight
        1.0,      # subsample
        5,        # subsample_freq
        1.0,      # colsample_bytree
        10.0,     # reg_alpha
        50.0,     # reg_lambda
        1.0,      # min_split_gain
        1024      # max_bin
    ]
    
    archive_size = pop_size
    H = 10
    MAX_FEV = 1500
    best_in_RUN = []
    fitness_stat = pd.DataFrame()
    history_tracking = []
    
    while (RUNS>0):
        df_stats = pd.DataFrame(columns = columns)
        
        global_best = 1e300
        population = np.empty((pop_size,N))
        v = np.empty((pop_size,N))
        archive = np.empty((archive_size,N))
        fitness = np.empty(pop_size)
        fitness_new = np.empty(pop_size)
        archive_fitness = np.empty(archive_size)
        archive_fitness.fill(np.inf)
    
        F_history = np.empty(H)
        CR_history = np.empty(H)
        S_CR = []
        S_F = []
        w = []
        r = np.empty(pop_size)
        CR = np.empty(pop_size)
        F = np.empty(pop_size)
        p = np.empty(pop_size)
    
        FEV = 1500
        normalized_prob = get_feature_generation_probabilities(TID=TID, plot=False)
        prob_aligned = normalized_prob.sort_index(
            key=lambda x: x.str.extract(r'(\d+)')[0].astype(int)
        )
    
        init_population(population, pop_size, N, a, b, variable_type, prob_aligned, n1)
    
        for i in range (0, pop_size):
    
            x = population[i][:]
            solution = x.copy()
    
            fit = fitness_lgbm_mae_all_features_HPO(cache, solution, n1, n2)
            fitness[i] = fit[1]
            FEV = FEV - 1
            
            if (fitness[i]<global_best):
                global_best = fitness[i]
                best_solution = solution.copy()
                best_fitness_ever = fit[0], fit[1], fit[2]
                print("RUN: ",MAX_RUNS-RUNS+1," FEVs: ", MAX_FEV - FEV, ": ", "MAE_val: ",best_fitness_ever[0], "fitness_penatly_log-based: ", best_fitness_ever[1], "features_n:", best_fitness_ever[2] )
    
                
    
            history_tracking = [
                MAX_FEV - FEV,
                best_fitness_ever[0],
                best_fitness_ever[1],
                best_fitness_ever[2],
                *best_solution]
    
    
            df_stats.loc[len(df_stats)] = history_tracking
            
            
        F_history[:] = 0.5
        CR_history[:] = 0.5
        A = 0
        k=0
        pmin = 5.0/pop_size
    
        while (FEV>0):
            CR_df = pd.DataFrame(pd.DataFrame(CR_history).T)
    
            S_CR = np.empty(pop_size)
            S_F = np.empty(pop_size)
            v[:][:]=population[:][:]
            for i in range (0, pop_size):
                r[i] = int(random.random()*H)
                CR[i] = np.random.normal(CR_history[int(r[i])], 0.1)
                if CR[i]>1:
                    CR[i] = 1
                if CR[i]<0:
                    CR[i] = 0
    
                F[i] = F_history[int(r[i])]+np.random.standard_cauchy()*0.1
                if F[i]>1:
                    F[i] = 1
                while (F[i] < 0):
                    F[i] = F_history[int(r[i])]+np.random.standard_cauchy()*0.1
                p[i] = random.random()*(0.2-pmin)+pmin
    
                r1, r2, pbest = generate_indices(pop_size, A, p, i)
    
                j_rand = int(random.random()*N)
                for j in range (0,n1):
                    if (random.random()<CR[i] or j == j_rand):
                        if (r2 < pop_size):
                            v[i][j] = population[i][j]+F[i]*(population[pbest][j]-population[i][j])+F[i]*(population[r1][j]-population[r2][j])                       
                        if (r2 >= pop_size):
                            r2_arch = r2 - pop_size
                            v[i][j] = population[i][j]+F[i]*(population[pbest][j]-population[i][j])+F[i]*(population[r1][j]-archive[r2_arch][j])

                for j in range(n1, n1 + n2):
                    if random.random() < CR[i] or j == j_rand:
                
                        candidates = [i, pbest, r1, r2]
                
                        cand_fitness = []
                        for idx in candidates:
                            if idx < pop_size:
                                cand_fitness.append(fitness[idx])
                            else:
                                cand_fitness.append(archive_fitness[idx - pop_size])
                
                        cand_fitness = np.array(cand_fitness, dtype=float)
                
                        order = np.argsort(cand_fitness)
                
                        ranks = np.empty(len(candidates), dtype=float)
                        for rank_pos, cand_pos in enumerate(order):
                            ranks[cand_pos] = len(candidates) - rank_pos
                
                        probs = ranks / ranks.sum()
                
                        rnd_bool_var = np.random.choice(candidates, p=probs)
                
                        if rnd_bool_var < pop_size:
                            donor = population[rnd_bool_var]
                        else:
                            donor = archive[rnd_bool_var - pop_size]
                
                        v[i][j] = donor[j]
                
                remaining_ratio = max(0.0, min(1.0, FEV / MAX_FEV))
                
                p_mut_start = 3.0 / n2
                p_mut_end = 0.3 / n2
                gamma = 2.0
                p_mut = p_mut_end + (p_mut_start - p_mut_end) * (remaining_ratio ** gamma)
                p_mut = np.clip(p_mut, 0.001, 0.2)
                if random.random() < p_mut:
                    rnd_ind = random.randint(n1, n1 + n2 - 1)
                    v[i][rnd_ind] = 1 - int(v[i][rnd_ind])
    
                for j in range (0,n1):
                    if ( (isNaN(v[i][j]) == 1)):
                        if variable_type[j] == 'real':
                            v[i][j] = random.uniform(a[j], b[j])
                        if variable_type[j] == 'int':
                            v[i][j] = random.randint(int(a[j]), int(b[j]))
    
                check_sum_ones = 0
                for j in range (n1,n1 + n2):
                    if (v[i][j] == 1):
                        check_sum_ones = check_sum_ones +1
                if (check_sum_ones == 0):
                    v[i][random.randint(n1, n1 + n2-1)] = 1
                borders (v, population, N, a, b, variable_type)
    
    
    
            w = np.array([])
            S_CR = np.array([])
            S_F = np.array([])
            for i in range (0, pop_size):
    
                solution = v[i][:].copy()
                fit = fitness_lgbm_mae_all_features_HPO(cache, solution, n1, n2)
                
                fitness_new[i] = fit[1]
                FEV = FEV-1
                if (fitness_new[i]<fitness[i]):
                    S_CR = np.append(S_CR, CR[i])
                    S_F = np.append(S_F, F[i])
                    w = np.append(w,(fitness[i] - fitness_new[i]))
    
                    if A < archive_size:
                        archive[A] = population[i][:]
                        archive_fitness[A] = fitness[i]
                        A += 1
                    else:
                        rnd_index = int(random.random() * archive_size)
                        archive[rnd_index] = population[i][:]
                        archive_fitness[rnd_index] = fitness[i]
                    A = min(A, archive_size)
                    population[i][:] = solution
    
                    fitness[i] = fitness_new[i]
    
    
                    if (fitness[i]<global_best):
                        global_best = fitness[i]
                        best_solution = solution.copy()
                        best_fitness_ever = fit[0], fit[1], fit[2]
                        print(TID, ":, RUN: ",MAX_RUNS-RUNS+1," FEVs: ", MAX_FEV - FEV, ": ", "MAE_val: ",best_fitness_ever[0], "fitness_penatly_log-based: ", best_fitness_ever[1], "features_n:", best_fitness_ever[2] )
                    
                history_tracking = [ MAX_FEV - FEV,  best_fitness_ever[0],
                best_fitness_ever[1], best_fitness_ever[2],
                *best_solution]
    
    
                df_stats.loc[len(df_stats)] = history_tracking
    
            total_w = np.sum(w)
            w = w/total_w
    
            new_CR = np.sum(w*S_CR)
            new_F = np.sum(w*S_F*S_F)/np.sum(w*S_F)
            if (new_CR >0 and new_F>0):
                
                F_history[k]=new_F
                CR_history[k]=new_CR
                k=k+1
                if (k>H-1):
                    k=0
        run_file = f"TID_{TID}_run_{MAX_RUNS-RUNS+1:02d}_history_LightGBM_new_GA-SHADE_0_005.csv"
        df_stats.to_csv(run_file, index=False)
        RUNS = RUNS - 1