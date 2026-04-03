"""
Train a GMM-VAE run from a config.json file.

Usage:
  python train.py <run_name>
  CUDA_VISIBLE_DEVICES=1 python train.py latent6_nodoy
"""
import sys
import os
import json
import time
import filelock
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ---- Model ----

class GMMVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, n_components):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components

        enc_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        dec_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
            prev_dim = h
        dec_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.mix_logits = nn.Parameter(torch.zeros(n_components))
        self.prior_mu = nn.Parameter(torch.randn(n_components, latent_dim) * 0.5)
        self.prior_logvar = nn.Parameter(torch.zeros(n_components, latent_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(z)

    def log_gmm_prior(self, z):
        z_exp = z.unsqueeze(1)
        mu = self.prior_mu.unsqueeze(0)
        logvar = self.prior_logvar.unsqueeze(0)
        log_p_k = -0.5 * (
            self.latent_dim * np.log(2 * np.pi)
            + logvar.sum(dim=-1)
            + ((z_exp - mu) ** 2 / logvar.exp()).sum(dim=-1)
        )
        log_pi = F.log_softmax(self.mix_logits, dim=0)
        return torch.logsumexp(log_pi + log_p_k, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss(self, x, x_recon, mu, logvar, z, beta=1.0, feature_weights=None):
        if feature_weights is not None:
            recon_loss = (feature_weights * (x_recon - x) ** 2).sum() / x.size(0)
        else:
            recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)
        log_q = -0.5 * (self.latent_dim * np.log(2 * np.pi)
                        + logvar.sum(dim=-1)
                        + ((z - mu) ** 2 / logvar.exp()).sum(dim=-1))
        log_p = self.log_gmm_prior(z)
        kl = (log_q - log_p).mean()
        return recon_loss + beta * kl, recon_loss, kl


def main():
    run_name = sys.argv[1]
    run_dir = f'runs/{run_name}'
    with open(f'{run_dir}/config.json') as f:
        cfg = json.load(f)
    os.makedirs(f'{run_dir}/plots', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{run_name}] Device: {device}", flush=True)

    K = cfg['K']
    LATENT_DIM = cfg['latent_dim']
    HIDDEN_DIMS = cfg['hidden_dims']
    BETA = cfg['beta']
    KL_WARMUP = cfg.get('kl_warmup_epochs', 0)
    BATCH_SIZE = cfg['batch_size']
    LR = cfg['lr']
    N_EPOCHS = cfg['epochs']
    PATIENCE = cfg['patience']
    EXCLUDE_DOY = cfg.get('exclude_doy', False)

    print(f"[{run_name}] K={K}, latent_dim={LATENT_DIM}, beta={BETA}, "
          f"kl_warmup={KL_WARMUP}, exclude_doy={EXCLUDE_DOY}", flush=True)

    # ---- Data ----
    data_path = '../svgp/full_conus/loso_temp_data.parquet'
    df = pd.read_parquet(data_path)

    time_varying = ['aot', 'wind', 'hgt', 'cld', 'longwave', 'rh', 'tmax', 'smogI', 'smogP']
    static = ['lat', 'lon', 'logpd2500g', 'minf_5000', 'sd50k',
              'heavy_industrial_ind1', 'housing']
    feature_cols = [f for f in time_varying if f in df.columns] \
                 + [f for f in static if f in df.columns]
    if not EXCLUDE_DOY:
        feature_cols += ['day_of_year']

    INPUT_DIM = len(feature_cols)
    print(f"[{run_name}] {len(df):,} obs, {INPUT_DIM} features", flush=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values).astype(np.float32)

    pm25 = df['pm25'].values
    seasons = pd.to_datetime(df['date'].values).month.map(
        lambda m: {12: 'Winter', 1: 'Winter', 2: 'Winter',
                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                   9: 'Fall', 10: 'Fall', 11: 'Fall'}[m]).values

    X_tensor = torch.tensor(X)
    n = len(X_tensor)
    n_val = int(0.1 * n)
    n_train = n - n_val
    train_ds, val_ds = random_split(TensorDataset(X_tensor), [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ---- Feature weights ----
    feat_wt = None
    if 'feature_weights' in cfg:
        feat_wt = torch.tensor(cfg['feature_weights'], dtype=torch.float32, device=device)
        print(f"[{run_name}] Feature weights: {cfg['feature_weights']}", flush=True)

    # ---- Model ----
    model = GMMVAE(INPUT_DIM, LATENT_DIM, HIDDEN_DIMS, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- Training ----
    history = {'train_loss': [], 'val_loss': [],
               'train_recon': [], 'val_recon': [],
               'train_kl': [], 'val_kl': [], 'beta_eff': []}
    best_val_loss = float('inf')
    patience_counter = 0
    t_start = time.perf_counter()

    for epoch in range(N_EPOCHS):
        beta_eff = BETA * min(1.0, (epoch + 1) / KL_WARMUP) if KL_WARMUP > 0 else BETA
        history['beta_eff'].append(beta_eff)

        model.train()
        ep_loss, ep_recon, ep_kl, nb = 0, 0, 0, 0
        for (bx,) in train_loader:
            bx = bx.to(device)
            xr, mu, lv, z = model(bx)
            loss, recon, kl = model.loss(bx, xr, mu, lv, z, beta=beta_eff, feature_weights=feat_wt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            ep_loss += loss.item(); ep_recon += recon.item(); ep_kl += kl.item(); nb += 1
        history['train_loss'].append(ep_loss/nb)
        history['train_recon'].append(ep_recon/nb)
        history['train_kl'].append(ep_kl/nb)

        model.eval()
        vl, vr, vk, nvb = 0, 0, 0, 0
        with torch.no_grad():
            for (bx,) in val_loader:
                bx = bx.to(device)
                xr, mu, lv, z = model(bx)
                loss, recon, kl = model.loss(bx, xr, mu, lv, z, beta=beta_eff, feature_weights=feat_wt)
                vl += loss.item(); vr += recon.item(); vk += kl.item(); nvb += 1
        history['val_loss'].append(vl/nvb)
        history['val_recon'].append(vr/nvb)
        history['val_kl'].append(vk/nvb)

        elapsed = time.perf_counter() - t_start
        print(f"[{run_name}] Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"train={history['train_loss'][-1]:.4f} (recon={history['train_recon'][-1]:.4f} kl={history['train_kl'][-1]:.4f}) | "
              f"val={history['val_loss'][-1]:.4f} | beta={beta_eff:.3f} | {elapsed:.0f}s", flush=True)

        if history['val_loss'][-1] < best_val_loss:
            best_val_loss = history['val_loss'][-1]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{run_name}] Early stopping at epoch {epoch+1}", flush=True)
                break

    model.load_state_dict(best_state); model.to(device)
    total_time = time.perf_counter() - t_start
    print(f"[{run_name}] Best val loss: {best_val_loss:.4f}, time: {total_time:.1f}s", flush=True)

    # ---- Encode ----
    model.eval()
    all_z, all_mu = [], []
    full_loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE*4, shuffle=False)
    with torch.no_grad():
        for (bx,) in full_loader:
            bx = bx.to(device)
            mu, lv = model.encode(bx)
            all_mu.append(mu.cpu()); all_z.append(model.reparameterize(mu, lv).cpu())
    Z = torch.cat(all_z).numpy()
    Z_mu = torch.cat(all_mu).numpy()

    mix_weights = F.softmax(model.mix_logits, dim=0).detach().cpu().numpy()
    prior_means = model.prior_mu.detach().cpu().numpy()
    prior_stds = torch.exp(0.5 * model.prior_logvar).detach().cpu().numpy()

    # ---- Reconstruction R² ----
    val_x_list, val_recon_list = [], []
    with torch.no_grad():
        for (bx,) in val_loader:
            bx = bx.to(device)
            xr, _, _, _ = model(bx)
            val_x_list.append(bx.cpu().numpy())
            val_recon_list.append(xr.cpu().numpy())
    x_orig = np.concatenate(val_x_list)
    x_recon = np.concatenate(val_recon_list)

    per_feat_mse = ((x_orig - x_recon)**2).mean(axis=0)
    per_feat_r2 = 1 - per_feat_mse / x_orig.var(axis=0)
    mean_feat_r2 = per_feat_r2.mean()
    ss_res = ((x_orig - x_recon)**2).sum()
    ss_tot = ((x_orig - x_orig.mean(axis=0))**2).sum()
    overall_r2 = 1 - ss_res / ss_tot

    # ---- Component assignments ----
    z_all = torch.tensor(Z_mu, dtype=torch.float32).unsqueeze(1)
    prior_mu_t = model.prior_mu.cpu().unsqueeze(0)
    prior_logvar_t = model.prior_logvar.cpu().unsqueeze(0)
    log_pi = F.log_softmax(model.mix_logits.cpu(), dim=0)
    assignments_list = []
    for start in range(0, len(z_all), 100_000):
        zc = z_all[start:start+100_000]
        log_p_k = -0.5 * (LATENT_DIM * np.log(2*np.pi)
                          + prior_logvar_t.sum(dim=-1)
                          + ((zc - prior_mu_t)**2 / prior_logvar_t.exp()).sum(dim=-1))
        assignments_list.append((log_pi + log_p_k).argmax(dim=1).numpy())
    assignments = np.concatenate(assignments_list)

    # ---- Plots ----
    np.random.seed(42)
    n_plot = 50_000
    pidx = np.random.choice(len(Z_mu), n_plot, replace=False)
    Zp = Z_mu[pidx]
    season_colors = {'Winter': '#1f77b4', 'Spring': '#2ca02c', 'Summer': '#ff7f0e', 'Fall': '#d62728'}

    # Latent space (first 2 dims)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    for s in ['Winter', 'Spring', 'Summer', 'Fall']:
        m = seasons[pidx] == s
        ax.scatter(Zp[m, 0], Zp[m, 1], c=season_colors[s], s=0.5, alpha=0.3, label=s, rasterized=True)
    ax.legend(markerscale=10, fontsize=9); ax.set_title('Season'); ax.set_xlabel('z1'); ax.set_ylabel('z2')

    ax = axes[1]
    sc = ax.scatter(Zp[:, 0], Zp[:, 1], c=np.log(pm25[pidx]+1), s=0.5, alpha=0.3, cmap='YlOrRd', rasterized=True)
    plt.colorbar(sc, ax=ax, label='log(PM2.5+1)'); ax.set_title('log(PM2.5)'); ax.set_xlabel('z1'); ax.set_ylabel('z2')

    ax = axes[2]
    ax.scatter(Zp[:, 0], Zp[:, 1], c='lightgray', s=0.3, alpha=0.2, rasterized=True)
    for k in range(K):
        if mix_weights[k] < 0.01: continue
        if LATENT_DIM >= 2:
            ell = Ellipse(xy=prior_means[k, :2], width=4*prior_stds[k,0], height=4*prior_stds[k,1],
                          fill=False, linewidth=2, edgecolor=f'C{k}', label=f'k={k} (pi={mix_weights[k]:.2f})')
            ax.add_patch(ell)
        ax.plot(prior_means[k,0], prior_means[k,1], 'x', color=f'C{k}', markersize=10, markeredgewidth=2)
    ax.legend(fontsize=7); ax.set_title('GMM Components'); ax.set_xlabel('z1'); ax.set_ylabel('z2')
    ax.set_xlim(Zp[:,0].min()-0.5, Zp[:,0].max()+0.5); ax.set_ylim(Zp[:,1].min()-0.5, Zp[:,1].max()+0.5)
    plt.tight_layout(); plt.savefig(f'{run_dir}/plots/latent_space.png', dpi=150, bbox_inches='tight'); plt.close()

    # Reconstruction
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(INPUT_DIM), per_feat_r2)
    ax.set_xticks(range(INPUT_DIM)); ax.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax.set_ylabel('Reconstruction R2'); ax.set_title(f'Per-Feature Reconstruction [{run_name}]')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5); ax.grid(True, alpha=0.3, axis='y')
    for i, (bar, r2) in enumerate(zip(bars, per_feat_r2)):
        bar.set_color('#2ca02c' if r2 > 0.5 else '#ff7f0e' if r2 > 0 else '#d62728')
        ax.text(i, r2 + 0.02, f'{r2:.2f}', ha='center', fontsize=7)
    plt.tight_layout(); plt.savefig(f'{run_dir}/plots/reconstruction.png', dpi=150, bbox_inches='tight'); plt.close()

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history['train_recon'], label='Train'); axes[1].plot(history['val_recon'], label='Val')
    axes[1].set_title('Reconstruction'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(history['train_kl'], label='Train KL'); axes[2].plot(history['val_kl'], label='Val KL')
    axes[2].set_title('KL Divergence'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    for ax in axes: ax.set_xlabel('Epoch')
    plt.tight_layout(); plt.savefig(f'{run_dir}/plots/loss_curves.png', dpi=150, bbox_inches='tight'); plt.close()

    # Component profiles
    unique, counts = np.unique(assignments, return_counts=True)
    profiles = pd.DataFrame(X, columns=feature_cols)
    profiles['component'] = assignments
    profiles = profiles.groupby('component')[feature_cols].mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(profiles.values, cmap='RdBu_r', aspect='auto', vmin=-1.5, vmax=1.5)
    ax.set_yticks(range(len(profiles))); ax.set_yticklabels([f'k={k} (n={c:,})' for k, c in zip(unique, counts)])
    ax.set_xticks(range(len(feature_cols))); ax.set_xticklabels(feature_cols, rotation=45, ha='right')
    ax.set_title(f'Component Profiles [{run_name}]'); plt.colorbar(im, ax=ax, label='Standardized value')
    for i in range(profiles.shape[0]):
        for j in range(profiles.shape[1]):
            v = profiles.iloc[i, j]
            if abs(v) > 0.5: ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6, fontweight='bold')
    plt.tight_layout(); plt.savefig(f'{run_dir}/plots/component_profiles.png', dpi=150, bbox_inches='tight'); plt.close()

    # ---- Save ----
    torch.save({
        'model_state_dict': model.state_dict(), 'config': cfg,
        'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
        'feature_cols': feature_cols, 'history': history, 'total_time': total_time,
    }, f'{run_dir}/model.pt')

    np.savez(f'{run_dir}/embeddings.npz', z_mu=Z_mu, z_sample=Z,
             assignments=assignments, mix_weights=mix_weights,
             prior_means=prior_means, prior_stds=prior_stds, feature_cols=feature_cols)

    # Update configs.csv (with file lock for parallel safety)
    lock = filelock.FileLock('configs.csv.lock', timeout=30)
    with lock:
        configs_df = pd.read_csv('configs.csv')
        result_row = {
            'name': run_name, 'K': K, 'latent_dim': LATENT_DIM, 'beta': BETA,
            'kl_warmup_epochs': KL_WARMUP, 'hidden_dims': ','.join(map(str, HIDDEN_DIMS)),
            'exclude_doy': EXCLUDE_DOY, 'batch_size': BATCH_SIZE, 'lr': LR,
            'epochs': N_EPOCHS, 'patience': PATIENCE, 'notes': cfg.get('notes', ''),
            'best_val_loss': f'{best_val_loss:.4f}',
            'final_recon': f'{history["val_recon"][-1]:.4f}',
            'final_kl': f'{history["val_kl"][-1]:.4f}',
            'active_components': int((mix_weights > 0.01).sum()),
            'mean_feat_r2': f'{mean_feat_r2:.4f}',
            'overall_r2': f'{overall_r2:.4f}',
            'train_time_s': f'{total_time:.0f}',
            'stopped_epoch': len(history['train_loss']),
        }
        if run_name in configs_df['name'].values:
            i = configs_df[configs_df['name'] == run_name].index[0]
            for col, val in result_row.items():
                configs_df.loc[i, col] = val
        else:
            configs_df = pd.concat([configs_df, pd.DataFrame([result_row])], ignore_index=True)
        configs_df.to_csv('configs.csv', index=False)

    print(f"[{run_name}] DONE: val_loss={best_val_loss:.4f}, mean_feat_r2={mean_feat_r2:.4f}, "
          f"overall_r2={overall_r2:.4f}, active_K={int((mix_weights > 0.01).sum())}/{K}", flush=True)


if __name__ == '__main__':
    main()
