from typing import Callable, Dict, List, Optional, Sequence, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import mean_squared_error, r2_score
from dataclasses import dataclass
from scipy.stats import spearmanr, pearsonr, kendalltau
from typing import List, Tuple, Dict, Any, Union
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from scipy.spatial.distance import cdist
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from hyppo.independence import Hsic
from openml.tasks import TaskType
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import random
import openml
import copy
import time
import os
import re

ROOT_DIR = "openml_regression_dump"
os.makedirs(ROOT_DIR, exist_ok=True)
    
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
                repeat=repeat, fold=fold, sample=0
            )

            split_folder = os.path.join(splits_dir, f"repeat_{repeat}_fold_{fold}")
            os.makedirs(split_folder, exist_ok=True)

            pd.DataFrame({"train_idx": train_idx}).to_csv(
                os.path.join(split_folder, "train_idx.csv"),
                index=False
            )
            pd.DataFrame({"test_idx": test_idx}).to_csv(
                os.path.join(split_folder, "test_idx.csv"),
                index=False
            )

    print(f"  -> saved to {task_dir}")


ROOT_DIR = "openml_regression_dump"


@dataclass(frozen=True)
class TaskCache:
    X: np.ndarray                       # shape (n, d)
    y: np.ndarray                       # shape (n,)
    splits: List[Tuple[np.ndarray, np.ndarray]]  # the list of (train_idx, test_idx)



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



def make_moead_config(
    pop_size: int = 100,
    max_fev: int = 5000,
    n_neighbors: int = 10,
    H: int = 10,
    seed: Optional[int] = None,
    bool_mutation_rate: Optional[float] = None,
    use_global_mating_probability: float = 0.0,
    nr: Optional[int] = None,
    verbose: bool = True,

    feature_generation_probabilities: Optional[Sequence[float]] = None,
    guided_initialization_fraction: float = 0.7,
) -> Dict[str, object]:
    return {
        "pop_size": pop_size,
        "max_fev": max_fev,
        "n_neighbors": n_neighbors,
        "H": H,
        "seed": seed,
        "bool_mutation_rate": bool_mutation_rate,
        "use_global_mating_probability": use_global_mating_probability,
        "nr": nr,
        "verbose": verbose,
        "feature_generation_probabilities": feature_generation_probabilities,
        "guided_initialization_fraction": guided_initialization_fraction,
    }

def make_weights_2obj(pop_size: int) -> np.ndarray:
    if pop_size < 2:
        raise ValueError("pop_size must be >= 2 for two-objective MOEA/D")

    weights = np.zeros((pop_size, 2), dtype=float)

    for i in range(pop_size):
        weights[i, 0] = i / (pop_size - 1)
        weights[i, 1] = 1.0 - weights[i, 0]

    return weights


def make_neighborhood(weights: np.ndarray, n_neighbors: int) -> np.ndarray:
    distances = cdist(weights, weights, metric="euclidean")
    return np.argsort(distances, axis=1)[:, :n_neighbors]


def tchebycheff_scalarization(
    objectives: np.ndarray,
    weight: np.ndarray,
    ideal: np.ndarray,
) -> float:
    eps = 1e-12
    return float(np.max(np.maximum(weight, eps) * np.abs(objectives - ideal)))


def repair_individual(
    x: np.ndarray,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    variable_type: Sequence[str],
    n_real_int: int,
    n_bool: int,
) -> np.ndarray:
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    n_var = n_real_int + n_bool

    for j in range(n_real_int):
        if np.isnan(x[j]) or np.isinf(x[j]):
            x[j] = random.uniform(lower_bounds[j], upper_bounds[j])

        if x[j] < lower_bounds[j]:
            x[j] = lower_bounds[j]
        elif x[j] > upper_bounds[j]:
            x[j] = upper_bounds[j]

        if variable_type[j] == "int":
            x[j] = int(round(x[j]))
            x[j] = min(max(x[j], int(lower_bounds[j])), int(upper_bounds[j]))

    for j in range(n_real_int, n_var):
        x[j] = 1.0 if x[j] >= 0.5 else 0.0

    if np.sum(x[n_real_int:]) == 0:#fix solution
        x[random.randint(n_real_int, n_var - 1)] = 1.0

    return x


def initialize_population(
    pop_size: int,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    variable_type: Sequence[str],
    n_real_int: int,
    n_bool: int,
    feature_generation_probabilities: Optional[Sequence[float]] = None,
    guided_initialization_fraction: float = 0.7,
) -> np.ndarray:
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    n_var = n_real_int + n_bool

    if len(variable_type) != n_var:
        raise ValueError("len(variable_type) must be equal to n_real_int + n_bool")

    if len(lower_bounds) != n_real_int or len(upper_bounds) != n_real_int:
        raise ValueError("Bounds are specified only for real/int variables")

    if feature_generation_probabilities is not None:
        feature_generation_probabilities = np.asarray(
            feature_generation_probabilities,
            dtype=float,
        )

        if len(feature_generation_probabilities) != n_bool:
            raise ValueError(
                "len(feature_generation_probabilities) must be equal to n_bool"
            )

        feature_generation_probabilities = np.clip(
            feature_generation_probabilities,
            0.0,
            1.0,
        )

    guided_initialization_fraction = float(
        np.clip(guided_initialization_fraction, 0.0, 1.0)
    )

    guided_count = int(round(pop_size * guided_initialization_fraction))

    population = np.zeros((pop_size, n_var), dtype=float)

    for i in range(pop_size):

        for j in range(n_real_int):
            if variable_type[j] == "real":
                population[i, j] = random.uniform(lower_bounds[j], upper_bounds[j])
            elif variable_type[j] == "int":
                population[i, j] = random.randint(
                    int(lower_bounds[j]),
                    int(upper_bounds[j]),
                )
            else:
                raise ValueError(
                    f"Unexpected variable type before bool part: {variable_type[j]}"
                )


        for j in range(n_real_int, n_var):
            bool_pos = j - n_real_int

            if (
                feature_generation_probabilities is not None
                and i < guided_count
            ):
                p = feature_generation_probabilities[bool_pos]
            else:
                p = 0.5

            population[i, j] = 1.0 if random.random() < p else 0.0


        if np.sum(population[i, n_real_int:]) == 0:
            if (
                feature_generation_probabilities is not None
                and i < guided_count
            ):
                best_bool_pos = int(np.argmax(feature_generation_probabilities))
                population[i, n_real_int + best_bool_pos] = 1.0
            else:
                random_bool_pos = random.randint(0, n_bool - 1)
                population[i, n_real_int + random_bool_pos] = 1.0

        population[i] = repair_individual(
            population[i],
            lower_bounds,
            upper_bounds,
            variable_type,
            n_real_int,
            n_bool,
        )

    return population

def sort_population_by_number_of_features(
    population: np.ndarray,
    n_real_int: int,
) -> np.ndarray:

    n_selected_features = np.sum(population[:, n_real_int:] >= 0.5, axis=1)
    sorted_indices = np.argsort(n_selected_features)
    return population[sorted_indices].copy()



def calc_initial_population_baseline_mae(raw_objectives: np.ndarray,) -> float: # Normalization based on initial population

    initial_mae = np.asarray(raw_objectives[:, 0], dtype=float)
    finite_mae = initial_mae[np.isfinite(initial_mae)]

    if len(finite_mae) == 0:
        return 1.0

    baseline_mae = float(np.median(finite_mae))

    if baseline_mae <= 0.0 or not np.isfinite(baseline_mae):
        baseline_mae = 1.0

    return baseline_mae


def normalize_objectives(raw_objectives: np.ndarray, baseline_mae: float, n_bool: int,) -> np.ndarray:

    raw_objectives = np.asarray(raw_objectives, dtype=float)

    if raw_objectives.ndim == 1:
        raw_objectives = raw_objectives.reshape(1, -1)
        return_one = True
    else:
        return_one = False

    normalized = np.zeros_like(raw_objectives, dtype=float)

    normalized[:, 0] = raw_objectives[:, 0] / float(baseline_mae)
    normalized[:, 1] = raw_objectives[:, 1] / float(n_bool)

    normalized[:, 0] = np.where(np.isnan(normalized[:, 0]), np.inf, normalized[:, 0])

    normalized[:, 1] = np.where(np.isnan(normalized[:, 1]), np.inf, normalized[:, 1], )

    if return_one:
        return normalized[0]

    return normalized



# F/CR adaptation


def sample_cr(memory_cr: np.ndarray, memory_index: int) -> float:
    cr = np.random.normal(memory_cr[memory_index], 0.1)
    return float(np.clip(cr, 0.0, 1.0))


def sample_f(memory_f: np.ndarray, memory_index: int) -> float:
    f = memory_f[memory_index] + 0.1 * np.random.standard_cauchy()
    attempts = 0

    while f <= 0.0 and attempts < 100:
        f = memory_f[memory_index] + 0.1 * np.random.standard_cauchy()
        attempts += 1

    if f <= 0.0:
        f = memory_f[memory_index]

    return float(min(f, 1.0))


def update_f_cr_memory(
    memory_f: np.ndarray,
    memory_cr: np.ndarray,
    memory_position: int,
    successful_f: List[float],
    successful_cr: List[float],
    improvements: List[float],
) -> int:
    if len(successful_f) == 0:
        return memory_position

    successful_f = np.asarray(successful_f, dtype=float)
    successful_cr = np.asarray(successful_cr, dtype=float)
    improvements = np.asarray(improvements, dtype=float)

    if np.sum(improvements) <= 0:
        weights = np.full_like(improvements, 1.0 / len(improvements), dtype=float)
    else:
        weights = improvements / np.sum(improvements)


    new_f = np.sum(weights * successful_f * successful_f) / max(
        np.sum(weights * successful_f),
        1e-12,
    )

    new_cr = np.sum(weights * successful_cr)

    memory_f[memory_position] = float(np.clip(new_f, 1e-12, 1.0))
    memory_cr[memory_position] = float(np.clip(new_cr, 0.0, 1.0))

    memory_position += 1

    if memory_position >= len(memory_f):
        memory_position = 0

    return memory_position


def choose_mating_pool(
    subproblem_index: int,
    neighborhood: np.ndarray,
    pop_size: int,
    use_global_mating_probability: float,
) -> np.ndarray:
    if random.random() < use_global_mating_probability:
        return np.arange(pop_size)

    return neighborhood[subproblem_index]


def make_offspring(
    population: np.ndarray,
    subproblem_index: int,
    mating_pool: np.ndarray,
    f: float,
    cr: float,
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    variable_type: Sequence[str],
    n_real_int: int,
    n_bool: int,
    bool_mutation_rate: float,
) -> np.ndarray:
    pop_size = population.shape[0]
    n_var = n_real_int + n_bool

    if len(mating_pool) < 3:
        candidates = np.arange(pop_size)
    else:
        candidates = mating_pool

    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

    parent = population[subproblem_index].copy()
    mutant = parent.copy()

    # Continuous/integer part: DE/rand/1.
    mutant[:n_real_int] = population[r1, :n_real_int] + f * (
        population[r2, :n_real_int] - population[r3, :n_real_int]
    )

    trial = parent.copy()
    j_rand = random.randint(0, n_real_int - 1)

    for j in range(n_real_int):
        if random.random() < cr or j == j_rand:
            trial[j] = mutant[j]

    donors = [r1, r2, r3]
    
    donors = [r1, r2, r3]
    j_bool_rand = random.randint(n_real_int, n_var - 1)
    
    for j in range(n_real_int, n_var):
        if random.random() < cr or j == j_bool_rand:
            donor = random.choice(donors)
            trial[j] = population[donor, j]
        else:
            trial[j] = parent[j]
    
        if random.random() < bool_mutation_rate:
            trial[j] = 1.0 - trial[j]

    return repair_individual(
        trial,
        lower_bounds,
        upper_bounds,
        variable_type,
        n_real_int,
        n_bool,
    )



# External nondominated archive for tracking


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def update_external_archive(
    population: np.ndarray,
    raw_objectives: np.ndarray,
    normalized_objectives: np.ndarray,
    archive_x: List[np.ndarray],
    archive_raw_f: List[np.ndarray],
    archive_norm_f: List[np.ndarray],
) -> None:


    candidate_x = archive_x + [x.copy() for x in population]
    candidate_raw_f = archive_raw_f + [f.copy() for f in raw_objectives]
    candidate_norm_f = archive_norm_f + [f.copy() for f in normalized_objectives]

    new_archive_x: List[np.ndarray] = []
    new_archive_raw_f: List[np.ndarray] = []
    new_archive_norm_f: List[np.ndarray] = []

    for i, f_i in enumerate(candidate_norm_f):
        dominated = False
        duplicate = False

        for kept_f in new_archive_norm_f:
            if np.allclose(f_i, kept_f, rtol=1e-10, atol=1e-12):
                duplicate = True
                break

        if duplicate:
            continue

        for j, f_j in enumerate(candidate_norm_f):
            if i != j and dominates(f_j, f_i):
                dominated = True
                break

        if not dominated:
            new_archive_x.append(candidate_x[i].copy())
            new_archive_raw_f.append(candidate_raw_f[i].copy())
            new_archive_norm_f.append(candidate_norm_f[i].copy())

    archive_x[:] = new_archive_x
    archive_raw_f[:] = new_archive_raw_f
    archive_norm_f[:] = new_archive_norm_f


# for saving raw front every N FEV to one CSV


def save_pareto_snapshot(
    archive_raw_f: List[np.ndarray],
    fev: int,
    generation: int,
    snapshot_path: str,
    run_meta: Dict[str, object],
) -> None:
    """
    Appends current raw Pareto front to a single CSV file.
    Each row = one archive solution at the given FEV checkpoint.

    Columns: fev_checkpoint, generation, <run_meta keys>,
             validation_MAE, n_selected
    """
    if len(archive_raw_f) == 0:
        return

    rows = []
    for raw_f in archive_raw_f:
        rows.append(
            {
                "fev_checkpoint": fev,
                "generation": generation,
                **run_meta,
                "validation_MAE": float(raw_f[0]),
                "n_selected": int(round(float(raw_f[1]))),
            }
        )

    df = pd.DataFrame(rows)

    write_header = not os.path.exists(snapshot_path)
    df.to_csv(snapshot_path, mode="a", header=write_header, index=False)



# GA-SHADE-MO-MOEA/D-based

def run_moead_adaptive_f_cr(
    objective_fn: Callable[[np.ndarray, Optional[int]], np.ndarray],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    variable_type: Sequence[str],
    n_real_int: int,
    n_bool: int,
    config: Dict[str, object],
    snapshot_path: Optional[str] = None,
    snapshot_interval: int = 100,
    run_meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    pop_size = int(config.get("pop_size", 100))
    max_fev = int(config.get("max_fev", 5000))
    n_neighbors = int(config.get("n_neighbors", 10))
    H = int(config.get("H", 10))

    seed = config.get("seed", None)

    use_global_mating_probability = float(
        config.get("use_global_mating_probability", 0.0)
    )

    nr = config.get("nr", None)
    verbose = bool(config.get("verbose", True))

    if nr is None:
        nr = n_neighbors

    nr = int(nr)

    bool_mutation_rate = config.get("bool_mutation_rate", None)

    if bool_mutation_rate is None:
        bool_mutation_rate = 1.0 / max(1, n_bool)

    bool_mutation_rate = float(bool_mutation_rate)
    feature_generation_probabilities = config.get(
        "feature_generation_probabilities",
        None,
    )

    guided_initialization_fraction = float(
        config.get("guided_initialization_fraction", 0.7)
    )

    if run_meta is None:
        run_meta = {}

    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)

    weights = make_weights_2obj(pop_size)
    neighborhood = make_neighborhood(weights, n_neighbors)

    population = initialize_population(
        pop_size,
        lower_bounds,
        upper_bounds,
        variable_type,
        n_real_int,
        n_bool,
        feature_generation_probabilities=feature_generation_probabilities,
        guided_initialization_fraction=guided_initialization_fraction,
    )

    population = sort_population_by_number_of_features(
        population,
        n_real_int,
    )

    raw_objectives = np.zeros((pop_size, 2), dtype=float)
    fev = 0

    for i in range(pop_size):
        raw_objectives[i] = np.asarray(
            objective_fn(population[i].copy(), i),
            dtype=float,
        )
        fev += 1

    baseline_mae = calc_initial_population_baseline_mae(raw_objectives)

    objectives = normalize_objectives(
        raw_objectives=raw_objectives,
        baseline_mae=baseline_mae,
        n_bool=n_bool,
    )

    ideal = np.min(objectives, axis=0)

    memory_f = np.full(H, 0.5, dtype=float)
    memory_cr = np.full(H, 0.5, dtype=float)
    memory_position = 0

    archive_x: List[np.ndarray] = []
    archive_raw_f: List[np.ndarray] = []
    archive_norm_f: List[np.ndarray] = []

    update_external_archive(
        population=population,
        raw_objectives=raw_objectives,
        normalized_objectives=objectives,
        archive_x=archive_x,
        archive_raw_f=archive_raw_f,
        archive_norm_f=archive_norm_f,
    )

    last_snapshot_fev = 0

    if snapshot_path is not None:
        save_pareto_snapshot(
            archive_raw_f=archive_raw_f,
            fev=fev,
            generation=0,
            snapshot_path=snapshot_path,
            run_meta=run_meta,
        )
        last_snapshot_fev = fev

    history = []
    generation = 0

    if verbose:
        print(
            f"initial_population_evaluated: fev={fev}, "
            f"baseline_MAE_from_initial_population={baseline_mae:.6f}"
        )

    while fev < max_fev:
        successful_f: List[float] = []
        successful_cr: List[float] = []
        improvements: List[float] = []

        subproblem_order = np.random.permutation(pop_size)

        for i in subproblem_order:
            if fev >= max_fev:
                break

            h_index = random.randint(0, H - 1)

            f = sample_f(memory_f, h_index)
            cr = sample_cr(memory_cr, h_index)

            mating_pool = choose_mating_pool(
                i,
                neighborhood,
                pop_size,
                use_global_mating_probability,
            )

            child = make_offspring(
                population,
                i,
                mating_pool,
                f,
                cr,
                lower_bounds,
                upper_bounds,
                variable_type,
                n_real_int,
                n_bool,
                bool_mutation_rate,
            )

            child_raw_objectives = np.asarray(
                objective_fn(child.copy(), int(i)),
                dtype=float,
            )

            fev += 1

            if (
                snapshot_path is not None
                and fev - last_snapshot_fev >= snapshot_interval
            ):
                update_external_archive(
                    population=population,
                    raw_objectives=raw_objectives,
                    normalized_objectives=objectives,
                    archive_x=archive_x,
                    archive_raw_f=archive_raw_f,
                    archive_norm_f=archive_norm_f,
                )
                save_pareto_snapshot(
                    archive_raw_f=archive_raw_f,
                    fev=fev,
                    generation=generation,
                    snapshot_path=snapshot_path,
                    run_meta=run_meta,
                )
                last_snapshot_fev = fev

            child_objectives = normalize_objectives(
                raw_objectives=child_raw_objectives,
                baseline_mae=baseline_mae,
                n_bool=n_bool,
            )

            ideal = np.minimum(ideal, child_objectives)

            replacement_pool = neighborhood[i].copy()
            np.random.shuffle(replacement_pool)

            replacements = 0
            total_improvement = 0.0

            for j in replacement_pool:
                old_scalar = tchebycheff_scalarization(
                    objectives[j],
                    weights[j],
                    ideal,
                )

                new_scalar = tchebycheff_scalarization(
                    child_objectives,
                    weights[j],
                    ideal,
                )

                if new_scalar <= old_scalar:
                    population[j] = child.copy()

                    raw_objectives[j] = child_raw_objectives.copy()
                    objectives[j] = child_objectives.copy()

                    replacements += 1
                    total_improvement += max(old_scalar - new_scalar, 0.0)

                    if replacements >= nr:
                        break

            if replacements > 0:
                successful_f.append(f)
                successful_cr.append(cr)
                improvements.append(
                    total_improvement if total_improvement > 0.0 else 1e-12
                )

        memory_position = update_f_cr_memory(
            memory_f,
            memory_cr,
            memory_position,
            successful_f,
            successful_cr,
            improvements,
        )

        update_external_archive(
            population=population,
            raw_objectives=raw_objectives,
            normalized_objectives=objectives,
            archive_x=archive_x,
            archive_raw_f=archive_raw_f,
            archive_norm_f=archive_norm_f,
        )

        generation += 1

        archive_raw_objectives = np.asarray(archive_raw_f)
        archive_norm_objectives = np.asarray(archive_norm_f)

        if len(archive_norm_objectives) > 0:
            best_mae_norm = float(np.min(archive_norm_objectives[:, 0]))
            best_selected_ratio = float(np.min(archive_norm_objectives[:, 1]))
            best_mae = float(np.min(archive_raw_objectives[:, 0]))
            best_n_selected = float(np.min(archive_raw_objectives[:, 1]))
        else:
            best_mae_norm = float(np.min(objectives[:, 0]))
            best_selected_ratio = float(np.min(objectives[:, 1]))
            best_mae = float(np.min(raw_objectives[:, 0]))
            best_n_selected = float(np.min(raw_objectives[:, 1]))

        if verbose:
            print(
                f"gen={generation:04d}, fev={fev:05d}, "
                f"archive={len(archive_x)}, "
                f"best_MAE={best_mae:.6f}, "
                f"best_n_selected={best_n_selected:.0f}, "
                f"best_MAE_norm={best_mae_norm:.6f}, "
                f"best_selected_ratio={best_selected_ratio:.6f}, "
                f"mean_F={np.mean(memory_f):.3f}, "
                f"mean_CR={np.mean(memory_cr):.3f}"
            )

        history.append(
            {
                "generation": generation,
                "fev": fev,
                "baseline_MAE_from_initial_population": float(baseline_mae),
                "ideal_MAE_norm": float(ideal[0]),
                "ideal_selected_ratio": float(ideal[1]),
                "best_MAE": float(best_mae),
                "best_n_selected": float(best_n_selected),
                "best_MAE_norm": float(best_mae_norm),
                "best_selected_ratio": float(best_selected_ratio),
                "mean_F": float(np.mean(memory_f)),
                "mean_CR": float(np.mean(memory_cr)),
                "archive_size": len(archive_x),
            }
        )


    if snapshot_path is not None and fev > last_snapshot_fev:
        save_pareto_snapshot(
            archive_raw_f=archive_raw_f,
            fev=fev,
            generation=generation,
            snapshot_path=snapshot_path,
            run_meta=run_meta,
        )

    return {
        "population": population,
        "objectives": objectives,
        "raw_objectives": raw_objectives,
        "weights": weights,
        "neighborhood": neighborhood,
        "ideal": ideal,
        "baseline_MAE": baseline_mae,
        "memory_F": memory_f,
        "memory_CR": memory_cr,
        "archive_X": np.asarray(archive_x),
        "archive_F_raw": np.asarray(archive_raw_f),
        "archive_F_norm": np.asarray(archive_norm_f),
        "history": pd.DataFrame(history),
    }


def make_lightgbm_feature_objective(
    cache,
    n1: int,
    n2: int,
    calc_criterions: Callable,
) -> Callable[[np.ndarray, Optional[int]], np.ndarray]:

    def objective(
        solution: np.ndarray,
        subproblem_index: Optional[int] = None,
    ) -> np.ndarray:
        performance, n_features = calc_criterions(
            cache=cache,
            solution=solution,
            n1=n1,
            n2=n2,
        )
        return np.asarray(
            [
                float(performance),
                float(n_features),
            ],
            dtype=float,
        )
    return objective

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
            safe_abs_corr(pearsonr, X.iloc[:, j], y) for j in range(n_features)
        ])

        scores["spearman_abs"] = np.array([
            safe_abs_corr(spearmanr, X.iloc[:, j], y) for j in range(n_features)
        ])

        scores["kendall_abs"] = np.array([
            safe_abs_corr(kendalltau, X.iloc[:, j], y) for j in range(n_features)
        ])

        try:
            mi = mutual_info_regression(X, np.asarray(y).ravel(), random_state=mi_random_state)
            mi = np.nan_to_num(mi, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            mi = np.zeros(n_features, dtype=float)
        scores["mi"] = mi

        y_arr = np.asarray(y).ravel()
        scores["dcor"] = np.array([
            distance_correlation(X.iloc[:, j].values, y_arr) for j in range(n_features)
        ])
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
        X_tr_scaled = pd.DataFrame(
            scaler.fit_transform(X_tr),
            columns=X_tr.columns,
            index=X_tr.index
        )

        fold_result = filter_ranking_regression(
            X_tr_scaled,
            y_tr.values,
            total_points=total_points,
            mi_random_state=mi_random_state + fold_id
        )

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

import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

def calc_criterions(cache, solution, n1, n2):
    solution = np.asarray(solution, dtype=float)

    if len(solution) < n1 + n2:
        raise ValueError(
            f"Solution length is too small: len(solution)={len(solution)}, "
            f"expected at least {n1 + n2}"
        )
    X = np.asarray(cache.X)
    y = np.asarray(cache.y)

    if n2 != X.shape[1]:
        raise ValueError(
            f"n2 must match number of features: n2={n2}, X.shape[1]={X.shape[1]}")

    hp = solution[:n1].copy()

    feature_mask = solution[n1:n1 + n2]
    feature_mask = (feature_mask >= 0.5).astype(int)

    n_features = int(feature_mask.sum())

    if n_features == 0:
        return float("inf"), 0

    feature_idx = np.flatnonzero(feature_mask)
    X = X[:, feature_idx]

    boosting_types = ['gbdt', 'dart', 'goss']
    bt_idx = int(hp[0])
    bt_idx = max(0, min(bt_idx, len(boosting_types) - 1))
    boosting_type = boosting_types[bt_idx]
    subsample_val = float(hp[7])
    if boosting_type == 'goss':
        subsample_val = 1.0

    model_params = dict(
        objective='regression_l1',
        boosting_type=boosting_type,
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
        verbosity=-1
    )
    scores = []

    callbacks = []
    if boosting_type != 'dart':
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False
            )
        ]

    for train_idx, valid_idx in cache.splits:
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        model = lgb.LGBMRegressor(**model_params)

        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric='l1',
                callbacks=callbacks
            )

            y_pred = model.predict(
                X_valid,
                validate_features=False
            )

            scores.append(mean_absolute_error(y_valid, y_pred))

        except Exception:
            return float("inf"), n_features

    performance = float(np.mean(scores))

    return performance, n_features
    
import time
for TID in [360069, 4917]:

    save_task_by_tid(TID)
    cache = load_saved_task_cache(TID)
    
    n1 = 14
    n2 = cache.X.shape[1]
    N = n1 + n2
    
    a = [
        0,        # boosting_type
        0.005,    # learning_rate
        200,      # n_estimators
        31,       # num_leaves
        2,        # max_depth
        10,       # min_child_samples
        1e-3,     # min_child_weight
        0.4,      # subsample
        0,        # subsample_freq
        0.4,      # colsample_bytree
        0.0,      # reg_alpha
        0.0,      # reg_lambda
        0.0,      # min_split_gain
        64,       # max_bin
    ]
    
    b = [
        2.99,     # boosting_type index
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
        1024,     # max_bin
    ]
    
    variable_type = ["real"] * n1 + ["bool"] * n2
    
    objective = make_lightgbm_feature_objective(
        cache=cache,
        n1=n1,
        n2=n2,
        calc_criterions=calc_criterions,
    )
      
    # independent runs
    
    n_runs = 15
    base_seed = 42
    
    snapshot_file = f"LightGBM_GA-SHADE-MO-FI_TID_{TID}_pareto_snapshots.csv"

    if os.path.exists(snapshot_file): #removing old files
        os.remove(snapshot_file)
    
    all_solutions = []
    all_objectives = []
    all_histories = []
    
    total_start = time.perf_counter()
    
    for run_id in range(n_runs):
        seed = base_seed + run_id
    
        print("=" * 80)
        print(f"TID={TID} | run_id={run_id + 1}/{n_runs} | seed={seed}")
        print("=" * 80)
    
        run_meta = {
            "TID": TID,
            "run_id": run_id,
            "seed": seed,
            "algorithm": "GA-SHADE-MO-FI",
        }
    
        run_start = time.perf_counter()
        normalized_prob = get_feature_generation_probabilities(
        TID=TID,
        plot=False,
        )
        
        prob_aligned = normalized_prob.sort_index(
            key=lambda x: x.str.extract(r"(\d+)")[0].astype(int)
        )
        
        prob_aligned = prob_aligned.values
        
            
        config = make_moead_config(
            pop_size=100,
            max_fev=1500,
            n_neighbors=7,
            H=10,
            seed=seed,
            nr=2,
            use_global_mating_probability=0.07,
            bool_mutation_rate = 1.0/n2,
            verbose=True,
            feature_generation_probabilities=prob_aligned,
            guided_initialization_fraction=0.7,
        )
    
        result = run_moead_adaptive_f_cr(
            objective_fn=objective,
            lower_bounds=a,
            upper_bounds=b,
            variable_type=variable_type,
            n_real_int=n1,
            n_bool=n2,
            config=config,
            snapshot_path=snapshot_file,
            snapshot_interval=100,
            run_meta=run_meta,
        )
    
        run_time = time.perf_counter() - run_start
    
        pareto_solutions = pd.DataFrame(result["archive_X"])
    
        pareto_solutions["TID"] = TID
        pareto_solutions["run_id"] = run_id
        pareto_solutions["seed"] = seed
        pareto_solutions["algorithm"] = "GA-SHADE-MO-FI"

    
        pareto_objectives_raw = pd.DataFrame(
            result["archive_F_raw"],
            columns=[
                "validation_MAE",
                "n_selected",
            ],
        )
    
        pareto_objectives_norm = pd.DataFrame(
            result["archive_F_norm"],
            columns=[
                "MAE_norm",
                "selected_ratio",
            ],
        )
    
        pareto_objectives = pd.concat(
            [
                pareto_objectives_raw,
                pareto_objectives_norm,
            ],
            axis=1,
        )
    
        pareto_objectives["n_selected"] = (
            pareto_objectives["n_selected"]
            .round()
            .astype(int)
        )
    
        pareto_objectives["TID"] = TID
        pareto_objectives["run_id"] = run_id
        pareto_objectives["seed"] = seed
        pareto_objectives["baseline_MAE"] = result["baseline_MAE"]
        pareto_objectives["algorithm"] = "GA-SHADE-MO-FI"
        pareto_objectives["run_time_sec"] = run_time
    
        pareto_objectives = pareto_objectives[
            [
                "TID",
                "run_id",
                "seed",
                "validation_MAE",
                "n_selected",
                "MAE_norm",
                "selected_ratio",
                "baseline_MAE",
                "algorithm",
                "run_time_sec",
            ]
        ]

        
 
    
        history = result["history"].copy() # History
    
        history["TID"] = TID
        history["run_id"] = run_id
        history["seed"] = seed
        history["algorithm"] = "GA-SHADE-MO-FI"
        history["run_time_sec"] = run_time
    
        all_solutions.append(pareto_solutions)
        all_objectives.append(pareto_objectives)
        all_histories.append(history)
    
        print(
            f"Finished run_id={run_id + 1}/{n_runs} | "
            f"seed={seed} | "
            f"archive_size={len(pareto_objectives)} | "
            f"time={run_time:.2f} sec"
        )
    
    # Save all runs
    all_solutions = pd.concat(all_solutions, ignore_index=True)
    all_objectives = pd.concat(all_objectives, ignore_index=True)
    all_histories = pd.concat(all_histories, ignore_index=True)
    
    total_time = time.perf_counter() - total_start
    
    all_objectives["total_experiment_time_sec"] = total_time
    all_histories["total_experiment_time_sec"] = total_time
    
    all_solutions.to_csv(
        f"LightGBM_GA-SHADE-MO-FI_TID_{TID}_solutions_all_runs.csv",
        index=False,
    )
    
    all_objectives.to_csv(
        f"LightGBM_GA-SHADE-MO-FI_TID_{TID}_objectives_all_runs.csv",
        index=False,
    )
    
    all_histories.to_csv(
        f"LightGBM_GA-SHADE-MO-FI_TID_{TID}_history_all_runs.csv",
        index=False,
    )
    
    print("=" * 80)
    print(f"Finished all independent runs for TID={TID}")
    print(f"n_runs={n_runs}")
    print(f"max_fev_per_run=1500")
    print(f"total_nominal_fev={n_runs * 1500}")
    print(f"total_time={total_time:.2f} sec")
    print(f"snapshots saved to: {snapshot_file}")
    print("=" * 80)