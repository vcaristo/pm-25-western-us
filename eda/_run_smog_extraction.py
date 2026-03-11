"""
Script equivalent of the smog_I.ipynb extraction cells.
Extracts smog intensity from daily rasters and creates updated CSV.
"""
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
from collections import defaultdict

# Load main data
print("Loading data...")
df = pd.read_csv('/home/vcaristo/pm_data/data/pm25_data_complete_2003-2021_nodups_051922.csv', low_memory=False)
locs = pd.read_csv('/home/vcaristo/pm_data/data/pm25_locs_nodups_051922.csv')

print(f"Main data: {len(df)} rows, {df['date'].nunique()} unique dates")
print(f"Locations: {len(locs)} sites")

na_count = ((df['smogI'] == 'NA') | df['smogI'].isna()).sum()
print(f"smogI NA (including string 'NA'): {na_count} / {len(df)} ({na_count/len(df)*100:.1f}%)")

# Merge locations to get lat/lon
locs_subset = locs[['ll_id', 'lon', 'lat']].drop_duplicates('ll_id')
df = df.merge(locs_subset, on='ll_id', how='left')
print(f"After merge - rows with lat/lon: {df['lat'].notna().sum()} / {len(df)}")

# Build date -> raster path lookup
smog_dir = Path('/home/vcaristo/pm_data/data/smogI_093021')
smog_files = sorted(smog_dir.glob('smog_intensity_cfsr_*_30min.tif'))
print(f"\nAvailable smogI rasters: {len(smog_files)}")

date_to_raster = {}
for f in smog_files:
    date_str = f.stem.split('_')[3]
    date_to_raster[date_str] = f

df_dates = set(df['date'].astype(str).unique())
raster_dates = set(date_to_raster.keys())
covered = df_dates & raster_dates
print(f"Dates in data: {len(df_dates)}")
print(f"Dates with rasters: {len(raster_dates)}")
print(f"Dates covered: {len(covered)}")
print(f"Dates missing rasters: {len(df_dates - raster_dates)}")

# Extract smog intensity - optimized approach
# Pre-build index: for each date, which row indices and what ll_ids
print("\nBuilding index...")
df['date_str'] = df['date'].astype(str)

# Get unique coordinates per ll_id
site_coords = df.groupby('ll_id')[['lon', 'lat']].first().to_dict('index')

# Build date -> list of (ll_id, row_indices)
date_site_index = defaultdict(list)
for date_str, group in df.groupby('date_str'):
    if date_str in raster_dates:
        for ll_id, sub in group.groupby('ll_id'):
            date_site_index[date_str].append((ll_id, sub.index.tolist()))

# Initialize new smogI
smogI_values = np.full(len(df), np.nan)

print(f"Processing {len(date_site_index)} dates...")
processed = 0
sampled_rows = 0

for date_str, sites in date_site_index.items():
    raster_path = date_to_raster[date_str]

    try:
        with rasterio.open(raster_path) as src:
            band = src.read(1)
            nodata = src.nodata

            for ll_id, row_indices in sites:
                coords = site_coords[ll_id]
                lon, lat = coords['lon'], coords['lat']

                row, col = src.index(lon, lat)

                if 0 <= row < band.shape[0] and 0 <= col < band.shape[1]:
                    val = band[row, col]
                    if nodata is not None and val == nodata:
                        continue
                    for idx in row_indices:
                        smogI_values[idx] = float(val)
                    sampled_rows += len(row_indices)
    except Exception as e:
        print(f"Error processing {date_str}: {e}")
        continue

    processed += 1
    if processed % 500 == 0:
        print(f"  Processed {processed}/{len(date_site_index)} dates, sampled {sampled_rows} rows so far")

print(f"\nDone! Sampled {sampled_rows} rows across {processed} dates")

# Replace smogI column
df['smogI'] = smogI_values
print(f"New smogI coverage: {df['smogI'].notna().sum()} / {len(df)} ({df['smogI'].notna().mean()*100:.1f}%)")

# Drop helper columns
df_out = df.drop(columns=['date_str', 'lon', 'lat'])

# Save
output_path = '/home/vcaristo/pm_data/data/pm25_data_complete_2003_2021_smogI_031026.csv'
df_out.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"Shape: {df_out.shape}")
print(f"\nsmogI summary stats:")
print(df_out['smogI'].describe())
