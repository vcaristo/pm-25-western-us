"""
Parallel LOSO CV for Seasonal Interaction Kernel across multiple GPUs.
Distributes folds round-robin across available GPUs.

Usage: python loso_cv_parallel.py [--n_gpus 8] [--max_train 10000] [--n_epochs 50]
"""
import sys
sys.path.insert(0, '../..')
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp

from timing_utils import TimingLogger

np.random.seed(42)
torch.manual_seed(42)


# ── Model definition ────────────────────────────────────────────────────────

class SeasonalInteractionGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
                 period_init=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        self.base_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=len(base_indices),
                active_dims=torch.tensor(base_indices)))

        self.aot_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([aot_idx]))
        self.aot_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.aot_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.aot_periodic.initialize(period_length=period_init)
        self.summer_kernel = gpytorch.kernels.ScaleKernel(
            self.aot_rbf * self.aot_periodic * self.aot_seasonal_rbf)

        self.smog_rbf = gpytorch.kernels.RBFKernel(
            ard_num_dims=2, active_dims=torch.tensor([smogI_idx, smogP_idx]))
        self.smog_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.smog_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.smog_periodic.initialize(period_length=period_init)
        self.winter_kernel = gpytorch.kernels.ScaleKernel(
            self.smog_rbf * self.smog_periodic * self.smog_seasonal_rbf)

        self.residual_periodic = gpytorch.kernels.PeriodicKernel(active_dims=torch.tensor([doy_idx]))
        self.residual_seasonal_rbf = gpytorch.kernels.RBFKernel(active_dims=torch.tensor([doy_idx]))
        if period_init is not None:
            self.residual_periodic.initialize(period_length=period_init)
        self.seasonal_kernel = gpytorch.kernels.ScaleKernel(
            self.residual_periodic * self.residual_seasonal_rbf)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = (self.base_kernel(x) + self.summer_kernel(x)
                   + self.winter_kernel(x) + self.seasonal_kernel(x))
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ── Single fold worker ──────────────────────────────────────────────────────

def run_fold(args):
    """Run a single LOSO fold on a specific GPU."""
    (fold_idx, held_out_site, X_train, y_train_raw, X_test, y_test_raw,
     base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
     max_train_size, n_epochs, gpu_id, patience) = args

    device = torch.device(f'cuda:{gpu_id}')
    torch.manual_seed(42)
    np.random.seed(42 + fold_idx)

    # Log transform target
    y_train = np.log(y_train_raw + 1)
    y_test = np.log(y_test_raw + 1)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    period_init = 365.25 / scaler.scale_[doy_idx]

    # Subsample
    n_train_actual = len(X_train_scaled)
    if len(X_train_scaled) > max_train_size:
        idx = np.random.choice(len(X_train_scaled), max_train_size, replace=False)
        X_train_scaled = X_train_scaled[idx]
        y_train = y_train[idx]
        n_train_actual = max_train_size

    # Convert to tensors
    train_x = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    train_y = torch.tensor(y_train, dtype=torch.float32).to(device)
    test_x = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

    # Train
    train_start = time.perf_counter()

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SeasonalInteractionGP(
        train_x, train_y, likelihood,
        base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx, period_init
    ).to(device)
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    losses = []
    best_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    stopped_epoch = n_epochs

    for i in range(n_epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_likelihood_state = {k: v.clone() for k, v in likelihood.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience > 0 and epochs_without_improvement >= patience:
            stopped_epoch = i + 1
            break

    # Restore best model state
    if best_state is not None:
        model.load_state_dict(best_state)
        likelihood.load_state_dict(best_likelihood_state)

    train_time = time.perf_counter() - train_start

    # Predict
    infer_start = time.perf_counter()
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(test_x))
        pred_mean = pred.mean.cpu().numpy()
        pred_var = pred.variance.cpu().numpy()
    infer_time = time.perf_counter() - infer_start

    # Kernel params
    params = {
        'fold': fold_idx,
        'site': held_out_site,
        'base_scale': model.base_kernel.outputscale.item(),
        'summer_scale': model.summer_kernel.outputscale.item(),
        'winter_scale': model.winter_kernel.outputscale.item(),
        'seasonal_scale': model.seasonal_kernel.outputscale.item(),
        'aot_period_days': model.aot_periodic.period_length.item() * scaler.scale_[doy_idx],
        'smog_period_days': model.smog_periodic.period_length.item() * scaler.scale_[doy_idx],
        'residual_period_days': model.residual_periodic.period_length.item() * scaler.scale_[doy_idx],
        'noise': likelihood.noise.item(),
    }

    # Site metrics
    site_rmse = np.sqrt(np.mean((pred_mean - y_test)**2))
    site_mae = np.mean(np.abs(pred_mean - y_test))
    ss_res = np.sum((y_test - pred_mean)**2)
    ss_tot = np.sum((y_test - np.mean(y_test))**2)
    site_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan

    del model, likelihood, train_x, train_y, test_x
    torch.cuda.empty_cache()

    return {
        'fold_idx': fold_idx,
        'site': held_out_site,
        'pred_mean': pred_mean,
        'pred_var': pred_var,
        'y_test': y_test,
        'losses': losses,
        'params': params,
        'metrics': {
            'site': held_out_site,
            'n_obs': len(y_test),
            'rmse_log': site_rmse,
            'mae_log': site_mae,
            'r2_log': site_r2,
        },
        'timing': {
            'fold': fold_idx,
            'site': held_out_site,
            'n_train': n_train_actual,
            'n_test': len(y_test),
            'train_time': train_time,
            'infer_time': infer_time,
            'fold_time': train_time + infer_time,
            'stopped_epoch': stopped_epoch,
        },
        'gpu_id': gpu_id,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--max_train', type=int, default=10000)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (0 to disable)')
    args = parser.parse_args()

    print(f"Configuration: n_gpus={args.n_gpus}, max_train={args.max_train}, "
          f"n_epochs={args.n_epochs}, patience={args.patience}")
    print(f"Available CUDA devices: {torch.cuda.device_count()}")

    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    print(f"Using {n_gpus} GPUs")

    # Load data
    pm_all = pd.read_csv("../../../data/pm25_data_complete_2003_2021_smogI_031026.csv", low_memory=False)
    pm_fixed = pd.read_csv('../../../eda/pm25_locs_with_states.csv')

    mt_sites = pm_fixed[pm_fixed['state'] == 'MT'].copy()
    mt_ll_ids = set(mt_sites['ll_id'].values)

    pm_all['date'] = pd.to_datetime(pm_all['date'], format='%Y%m%d')
    pm_all['year'] = pm_all['date'].dt.year
    pm_mt = pm_all[(pm_all['ll_id'].isin(mt_ll_ids)) & (pm_all['year'].isin([2018, 2019]))].copy()

    time_varying_features = ['aot', 'wind', 'hgt', 'cld', 'longwave', 'rh', 'tmax', 'smogI', 'smogP']
    static_features = ['lat', 'lon', 'logpd2500g', 'minf_5000', 'sd50k',
                       'heavy_industrial_ind1', 'housing']

    available_tv = [f for f in time_varying_features if f in pm_mt.columns]
    available_static = [f for f in static_features if f in mt_sites.columns]

    pm_mt_subset = pm_mt[['ll_id', 'date', 'pm25'] + available_tv].copy()
    mt_static = mt_sites[['ll_id'] + available_static].copy()
    df = pm_mt_subset.merge(mt_static, on='ll_id', how='left')
    df['day_of_year'] = df['date'].dt.dayofyear

    feature_cols = available_tv + available_static + ['day_of_year']
    aot_idx = feature_cols.index('aot')
    smogI_idx = feature_cols.index('smogI')
    smogP_idx = feature_cols.index('smogP')
    doy_idx = feature_cols.index('day_of_year')
    seasonal_interaction_features = {'aot', 'smogI', 'smogP', 'day_of_year'}
    base_indices = [i for i, f in enumerate(feature_cols) if f not in seasonal_interaction_features]

    df_clean = df.dropna(subset=feature_cols + ['pm25']).copy()
    sites = df_clean['ll_id'].unique()
    print(f"\n{len(df_clean):,} observations, {len(sites)} sites")

    # Prepare fold arguments
    fold_args = []
    for i, held_out_site in enumerate(sites):
        test_mask = df_clean['ll_id'] == held_out_site
        train_df = df_clean[~test_mask]
        test_df = df_clean[test_mask]
        if len(test_df) == 0:
            continue

        X_train = train_df[feature_cols].values
        y_train_raw = train_df['pm25'].values
        X_test = test_df[feature_cols].values
        y_test_raw = test_df['pm25'].values
        gpu_id = i % n_gpus

        fold_args.append((
            i, held_out_site, X_train, y_train_raw, X_test, y_test_raw,
            base_indices, aot_idx, smogI_idx, smogP_idx, doy_idx,
            args.max_train, args.n_epochs, gpu_id, args.patience
        ))

    # Run folds in parallel
    print(f"\nLaunching {len(fold_args)} folds across {n_gpus} GPUs...")
    cv_start = time.perf_counter()

    results = []
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(run_fold, fa): fa[1] for fa in fold_args}
        for future in as_completed(futures):
            site = futures[future]
            try:
                result = future.result()
                results.append(result)
                r2 = result['metrics']['r2_log']
                gpu = result['gpu_id']
                ep = result['timing']['stopped_epoch']
                print(f"  Fold {result['fold_idx']:2d} (GPU {gpu}) site={site}: "
                      f"R²={r2:.3f}, epochs={ep}, train={result['timing']['train_time']:.1f}s")
            except Exception as e:
                print(f"  Fold for site {site} FAILED: {e}")

    cv_total_time = time.perf_counter() - cv_start
    print(f"\nTotal CV time: {cv_total_time:.1f}s ({cv_total_time/60:.1f} min)")

    # Sort results by fold index
    results.sort(key=lambda r: r['fold_idx'])

    # Aggregate predictions
    all_predictions = np.concatenate([r['pred_mean'] for r in results])
    all_actuals = np.concatenate([r['y_test'] for r in results])

    # Log scale metrics
    rmse_log = np.sqrt(np.mean((all_predictions - all_actuals)**2))
    mae_log = np.mean(np.abs(all_predictions - all_actuals))
    ss_res = np.sum((all_actuals - all_predictions)**2)
    ss_tot = np.sum((all_actuals - np.mean(all_actuals))**2)
    r2_log = 1 - (ss_res / ss_tot)

    # Original scale
    pred_pm25 = np.exp(all_predictions) - 1
    actual_pm25 = np.exp(all_actuals) - 1
    rmse_orig = np.sqrt(np.mean((pred_pm25 - actual_pm25)**2))
    mae_orig = np.mean(np.abs(pred_pm25 - actual_pm25))
    ss_res_orig = np.sum((actual_pm25 - pred_pm25)**2)
    ss_tot_orig = np.sum((actual_pm25 - np.mean(actual_pm25))**2)
    r2_orig = 1 - (ss_res_orig / ss_tot_orig)

    print(f"\n{'='*60}")
    print(f"LOSO CV Results (Seasonal Interaction, {n_gpus} GPUs)")
    print(f"MAX_TRAIN_SIZE={args.max_train}, N_EPOCHS={args.n_epochs}")
    print(f"{'='*60}")
    print(f"Log scale:  RMSE={rmse_log:.4f}, MAE={mae_log:.4f}, R²={r2_log:.4f}")
    print(f"Orig scale: RMSE={rmse_orig:.2f}, MAE={mae_orig:.2f}, R²={r2_orig:.4f}")
    print(f"Wall time:  {cv_total_time:.1f}s ({cv_total_time/60:.1f} min)")

    # Save results
    metrics_df = pd.DataFrame([r['metrics'] for r in results])
    params_df = pd.DataFrame([r['params'] for r in results])
    timing_df = pd.DataFrame([r['timing'] for r in results])

    # Save losses for plotting
    fold_losses = {r['site']: r['losses'] for r in results}

    output = {
        'config': {'max_train': args.max_train, 'n_epochs': args.n_epochs, 'n_gpus': n_gpus},
        'overall': {
            'rmse_log': rmse_log, 'mae_log': mae_log, 'r2_log': r2_log,
            'rmse_orig': rmse_orig, 'mae_orig': mae_orig, 'r2_orig': r2_orig,
            'total_time': cv_total_time,
        },
        'fold_losses': {site: losses for site, losses in fold_losses.items()},
    }

    with open('parallel_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    metrics_df.to_csv('parallel_site_metrics.csv', index=False)
    params_df.to_csv('parallel_kernel_params.csv', index=False)
    timing_df.to_csv('parallel_fold_timings.csv', index=False)

    print("\nResults saved to parallel_results.json, parallel_site_metrics.csv, etc.")

    # Print early stopping and loss summary
    stopped_epochs = [r['timing']['stopped_epoch'] for r in results]
    print(f"\nEarly stopping summary (patience={args.patience}):")
    print(f"  Stopped epochs: min={min(stopped_epochs)}, max={max(stopped_epochs)}, "
          f"mean={np.mean(stopped_epochs):.1f}")

    # Pad loss arrays to same length for averaging (folds may stop at different epochs)
    max_len = max(len(v) for v in fold_losses.values())
    loss_matrix = np.full((len(fold_losses), max_len), np.nan)
    for i, losses in enumerate(fold_losses.values()):
        loss_matrix[i, :len(losses)] = losses
    mean_loss = np.nanmean(loss_matrix, axis=0)
    min_epoch = np.nanargmin(mean_loss) + 1
    print(f"  Mean loss at epoch 1: {mean_loss[0]:.4f}")
    print(f"  Minimum mean loss at epoch {min_epoch}: {mean_loss[min_epoch-1]:.4f}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
