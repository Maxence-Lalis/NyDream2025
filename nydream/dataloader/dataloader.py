import torch
from torch.utils.data import Dataset

import ast
import os
from typing import Callable, List, Union, Iterable, Union, Optional, Tuple
import numpy as np
import pandas as pd

from dataloader.representation import molecular_graphs, molecular_graphs_with_ec50_rep
from dataloader.ec50_encoder import EC50EncoderV2
from dataloader.dataloader_utils import _safe_literal_eval,_ensure_2d_labels, _iterative_split, stratified_distance_split_graph # type: ignore

class DreamDataset(Dataset):
    def __init__(self, features, labels, cids, smiles, ids, concs):
        self.features = features
        self.labels = labels
        self.cids = cids
        self.smiles = smiles
        self.ids = ids
        self.concs = concs

        self._y_transform = None      
        self._y_inverse = None   

    def featurize(self, representation: Union[str, Callable], **kwargs):
        if isinstance(representation, str):
            if representation == "molecular_graphs":
                self.features = molecular_graphs(smiles=self.features, **kwargs)
            
            elif representation == "molecular_graphs_with_ec50_rep":
                ec50_df = pd.read_parquet('/home/maxence/Documents/temp/NyDREAM2025/nydream/rawdata/full_dream_all_parameters.parquet')
                ENCODER = EC50EncoderV2("/home/maxence/Documents/temp/NyDREAM2025/nydream/rawdata/best_set_autoenc3.pt", "cpu")
                self.features = molecular_graphs_with_ec50_rep(smiles=self.features, concs=self.concs, cids=self.cids, ec50_df=ec50_df, ENCODER=ENCODER, **kwargs)
            else:
                raise ValueError(f"Unknown representation '{representation}'")
        else:
            raise TypeError("representation must be a string or callable")
        
    def save(self, out_dir):
        if self.labels is None:
            labels_serialized = [""] * len(self.ids)
        else:
            labels_serialized = [
                str(row.tolist()) if hasattr(row, "tolist") else str(row)
                for row in self.labels
            ]

        return pd.DataFrame({
            "ID": self.ids,
            "SMILES": self.smiles,
            "CID": self.cids,
            "CONC": self.concs,
            "LABEL": labels_serialized,}).to_csv(out_dir)
    


    def set_label_transform(self, transform=None, inverse=None):
        self._y_transform = transform
        self._y_inverse = inverse
            
    def __len__(self):
        return len(self.features)

    def get_label_inverse(self):
        return self._y_inverse

    def __getitem__(self, idx: int):
        x = self.features[idx]
        if self.labels is None:
            return x
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self._y_transform is not None:
            y = self._y_transform(y)

        try:
            if hasattr(x, 'clone'):
                x = x.clone()
            if hasattr(x, 'y'):
                x.y = y
        except Exception:
            pass
        #print(x,y)
        return x, y


class DreamLoader:
    """
    Loads and cleans up your data
    """

    def __init__(
        self,
        smiles_column: str,
        label_column: str,
        id_column: str,
        conc_column: str,
        save_dir: str,
        sep: str = ",",
        mode: str = 'train',
        full_data_path: Optional[str] = None,
        train_path: Optional[str] = None,
        valid_path: Optional[str] = None,
        test_path: Optional[str] = None,
        exclude_path: Optional[str] = None,

        normalize_labels: bool = False,
        norm_method: str = "minmax",          # "minmax" | "standard"
        feature_range: Tuple[float, float] = (0.0, 1.0),  # only for minmax
        scaler_filename: str = "label_scaler.npz",
    ):
        self.smiles_column = smiles_column
        self.label_column = label_column
        self.id_column = id_column
        self.conc = conc_column
        self.sep = sep
        self.full_data_path = full_data_path
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.exclude_path = exclude_path 

        self.df: Optional[pd.DataFrame] = None
        self.train_ds: Optional[DreamDataset] = None
        self.valid_ds: Optional[DreamDataset] = None
        self.test_ds: Optional[DreamDataset] = None
        
        self.save_dir = save_dir

        self.normalize_labels = normalize_labels
        self.norm_method = norm_method.lower()
        self.feature_range = tuple(feature_range)
        self.scaler_path = os.path.join(self.save_dir, scaler_filename)

        self._scaler_fitted = False
        self._label_stats = {}  



    def _load(self, path, allow_missing_label: bool = False):
        df = pd.read_csv(path, sep=self.sep)
        assert self.id_column in df.columns, f"{self.id_column} not in CSV"
        assert self.smiles_column in df.columns, f"{self.smiles_column} not in CSV"
        
        if df[self.smiles_column].isna().any():
            print("[training] NaN in SMILES column. Removed.")
            df = df.dropna(subset=[self.smiles_column])
        
        if not allow_missing_label:
            assert self.label_column in df.columns, f"{self.label_column} not in CSV"
            labels = _ensure_2d_labels(df[self.label_column].values)
        else:
            labels = None
            
        df = df.set_index(self.id_column, drop=False)
        assert df.index.is_unique, f"{self.id_column} must be unique (1 row per ID)."



        has_cid = "CID" in df.columns
        if has_cid:
            df["CID"] = df["CID"].apply(_safe_literal_eval)
            df["CID"] = df["CID"].astype(int)
        
        has_conc = self.conc in df.columns
        if has_conc:
            df[self.conc] = df[self.conc].apply(_safe_literal_eval)
            df[self.conc] = df[self.conc].astype(float)
        
        smiles = df[self.smiles_column].astype(str).values
        ids = df[self.id_column].values
        cids = df["CID"].values if has_cid else np.array([None] * len(df), dtype=object)
        concs = df[self.conc].values if has_conc else np.array([0] * len(df), dtype=object)

        self.df = df
        return smiles, labels, cids, ids, concs

    def _load_exclude_ids(self, path: str) -> np.ndarray:
        ex = pd.read_csv(path, sep=self.sep)
        assert self.id_column in ex.columns, f"[exclude] {self.id_column} not in exclude CSV"
        return ex[self.id_column].astype(self.df[self.id_column].dtype if self.df is not None else ex[self.id_column].dtype).values

    def _make_dataset(self, smiles, labels, cids, ids, concs) -> DreamDataset:
        return DreamDataset(
            features=smiles,
            labels=labels,
            cids=cids,
            smiles=smiles,
            ids=ids,
            concs=concs
        )

    def split(
        self,
        valid_size: float = 0.2,
        test_size: float = 0.0,
        strategy: str = "iterative",         # "iterative" | "dup_stratified"
        dup_test_frac: float = 0.70,         # used when strategy="dup_stratified"
        dup_n_bins: int = 10,
        dup_seed: int = 0,
        ):
        """
        If self.exclude_path is provided:
            - All rows whose ID appears in that CSV go to the test set.
            - Remaining rows are split into train/valid with valid_size.
            - test_size is ignored.

        strategy:
            - "iterative"      : use existing _iterative_split (default).
            - "dup_stratified" : build a duplicate-aware TEST set using
                                 stratified_distance_split_graph; then split
                                 remaining rows into train/valid iteratively.
        """
        assert self.full_data_path is not None, "full_data_path must be provided for split()."
        smiles, labels, cids, ids, concs = self._load(self.full_data_path)

        if self.exclude_path is not None:
            exclude_ids = self._load_exclude_ids(self.exclude_path)
            exclude_mask = np.isin(ids, exclude_ids)
            missing = set(exclude_ids) - set(ids)
            print("[exclude] Number of excluded mols: ", len(exclude_ids))
            if len(missing) > 0:
                print(f"[exclude] Warning: {len(missing)} IDs in exclude file not found in full_data. Examples: {list(missing)[:5]}")

            test_idx = np.where(exclude_mask)[0]
            self.test_ds = self._make_dataset(smiles[test_idx],
                                              labels[test_idx] if labels is not None else None,
                                              cids[test_idx],
                                              ids[test_idx],
                                              concs[test_idx])
            keep_idx = np.where(~exclude_mask)[0]
            if keep_idx.size == 0:
                raise ValueError("[exclude] After excluding, no data left for train/valid.")

            sub_labels = labels[keep_idx]
            tr_sub_idx, val_sub_idx, _ = _iterative_split(
                sub_labels, valid_size=valid_size, test_size=0.0
            )
            train_idx = keep_idx[tr_sub_idx]
            valid_idx = keep_idx[val_sub_idx]

            self.train_ds = self._make_dataset(smiles[train_idx], labels[train_idx], cids[train_idx], ids[train_idx], concs[train_idx])
            self.valid_ds = self._make_dataset(smiles[valid_idx], labels[valid_idx], cids[valid_idx], ids[valid_idx], concs[valid_idx])
            if self.normalize_labels:
                self._fit_label_scaler()
                self._apply_label_scaler()
                self._save_label_scaler()
            return self.train_ds, self.valid_ds, self.test_ds

        if strategy == "dup_stratified" and (cids is not None) and (np.array(cids).dtype != object):
            labels_t = torch.tensor(labels, dtype=torch.float32)
            cids_t   = torch.tensor(cids, dtype=torch.long)

            train_pool_idx, valid_idx = stratified_distance_split_graph(
                labels=labels_t,
                cids=cids_t,
                test_frac=dup_test_frac,
                n_bins=dup_n_bins,
                seed=dup_seed,
            )
            train_idx = np.array(train_pool_idx, dtype=int)
            valid_idx = np.array(valid_idx, dtype=int)
            test_idx = np.array([], dtype=int)
            if test_size > 0:
                test_idx = np.random.choice(valid_idx, size=len(valid_idx)*test_size, replace=False)
                valid_idx = valid_idx[~test_idx]

            self.train_ds = self._make_dataset(smiles[train_idx], labels[train_idx], cids[train_idx], ids[train_idx], concs[train_idx])
            self.valid_ds = self._make_dataset(smiles[valid_idx], labels[valid_idx], cids[valid_idx], ids[valid_idx], concs[valid_idx])
            self.test_ds  = self._make_dataset(smiles[test_idx],  labels[test_idx],  cids[test_idx],  ids[test_idx],  concs[test_idx])
            if self.normalize_labels:
                self._fit_label_scaler()
                self._apply_label_scaler()
                self._save_label_scaler()
            
            return self.train_ds, self.valid_ds, self.test_ds

        train_idx, valid_idx, test_idx = _iterative_split(
            labels, valid_size=valid_size, test_size=test_size
        )
        self.train_ds = self._make_dataset(smiles[train_idx], labels[train_idx], cids[train_idx], ids[train_idx], concs[train_idx])
        self.valid_ds = self._make_dataset(smiles[valid_idx], labels[valid_idx], cids[valid_idx], ids[valid_idx], concs[valid_idx])
        self.test_ds  = self._make_dataset(smiles[test_idx],  labels[test_idx],  cids[test_idx],  ids[test_idx],  concs[test_idx])
        if self.normalize_labels:
            self._fit_label_scaler()
            self._apply_label_scaler()
            self._save_label_scaler()
        return self.train_ds, self.valid_ds, self.test_ds
        
    def load_split(self) -> Tuple[DreamDataset, Optional[DreamDataset], Optional[DreamDataset]]:
        """
        Load user-provided train/valid/test CSVs. The label column is optional for the test set.
        Provide any combination; missing valid or test will return None for that dataset.
        """
        assert self.train_path is not None, "[training] train_path must be provided for pre-split mode"

        t_smiles, t_labels, t_cids, t_ids, t_concs = self._load(self.train_path, allow_missing_label=False)
        self.train_ds = self._make_dataset(t_smiles, t_labels, t_cids, t_ids, t_concs)

        if self.valid_path is not None and os.path.exists(self.valid_path):
            v_smiles, v_labels, v_cids, v_ids, v_concs = self._load(self.valid_path, allow_missing_label=False)
            self.valid_ds = self._make_dataset(v_smiles, v_labels, v_cids, v_ids, v_concs)
        else:
            self.valid_ds = None

        if self.test_path is not None and os.path.exists(self.test_path):
            s_smiles, s_labels, s_cids, s_ids, v_concs = self._load(self.test_path, allow_missing_label=True)
            self.test_ds = self._make_dataset(s_smiles, s_labels, s_cids, s_ids, v_concs)
        else:
            self.test_ds = None

        if self.normalize_labels:
            self._fit_label_scaler()
            self._apply_label_scaler()
            self._save_label_scaler()
        return self.train_ds, self.valid_ds, self.test_ds


    def load_inference(self, path: Optional[str] = None) -> DreamDataset:
        """
        Load a CSV for inference (no labels required).
        Columns required: [id_column, smiles_column]. Optional: CID.
        """
        if path is None:
            assert self.test_path is not None, "Provide a CSV path or set test_path in __init__."
            path = self.test_path

        s_smiles, s_labels, s_cids, s_ids, s_concs = self._load(path, allow_missing_label=True)
        ds = self._make_dataset(s_smiles, s_labels, s_cids, s_ids, s_concs)
        self.test_ds = ds
        if self.normalize_labels:
            if not self._scaler_fitted:
                self._try_load_label_scaler()
            if self._scaler_fitted:
                fwd, inv = self._make_label_transforms()
                ds.set_label_transform(fwd, inv)
        self.test_ds = ds
        return ds

    def load_valid_only(self, path: Optional[str] = None) -> DreamDataset:
        """
        Load a single CSV that *has labels* for evaluation only.
        Required columns: [id_column, smiles_column, label_column]. Optional: CID.
        Returns the created DreamDataset and sets self.valid_ds.
        """
        if path is None:
            assert self.valid_path is not None and os.path.exists(self.valid_path), \
                "Provide a CSV path or set valid_path in __init__."
            path = self.valid_path

        v_smiles, v_labels, v_cids, v_ids, v_concs = self._load(path, allow_missing_label=False)
        ds = self._make_dataset(v_smiles, v_labels, v_cids, v_ids, v_concs)
        self.valid_ds = ds
        if self.normalize_labels:
            if not self._scaler_fitted:
                self._try_load_label_scaler()
            if self._scaler_fitted:
                fwd, inv = self._make_label_transforms()
                ds.set_label_transform(fwd, inv)
        self.valid_ds = ds
        return ds
    
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.train_ds is not None:
            self.train_ds.save(os.path.join(self.save_dir, "train.csv"))
        if self.valid_ds is not None:
            self.valid_ds.save(os.path.join(self.save_dir, "valid.csv"))
        if self.test_ds is not None:
            self.test_ds.save(os.path.join(self.save_dir, "test.csv"))
    
    def _fit_label_scaler(self, eps: float = 1e-8):
        assert self.train_ds is not None and self.train_ds.labels is not None, \
            "Call split()/load_split() first; train labels required to fit scaler."
        labels = self.train_ds.labels.astype(np.float32)  # (N, D)

        if self.norm_method == "minmax":
            lo = np.nanmin(labels, axis=0)
            hi = np.nanmax(labels, axis=0)
            same = (hi - lo) < eps
            hi[same] = lo[same] + 1.0
            a, b = map(float, self.feature_range)
            self._label_stats = {
                "method": "minmax",
                "min": lo, "max": hi,
                "a": a, "b": b,
            }

        elif self.norm_method == "standard":
            mu = np.nanmean(labels, axis=0)
            sd = np.nanstd(labels, axis=0)
            sd[sd < eps] = 1.0
            self._label_stats = {
                "method": "standard",
                "mean": mu, "std": sd,
            }

        else:
            raise ValueError(f"Unknown norm_method='{self.norm_method}'")

        self._scaler_fitted = True

    def _make_label_transforms(self):
        assert self._scaler_fitted and self._label_stats, "Fit scaler first."

        method = self._label_stats["method"]
        if method == "minmax":
            lo_t = torch.tensor(self._label_stats["min"], dtype=torch.float32)
            hi_t = torch.tensor(self._label_stats["max"], dtype=torch.float32)
            a = float(self._label_stats["a"]); b = float(self._label_stats["b"])

            def fwd(y: torch.Tensor):
                return (y - lo_t) / (hi_t - lo_t) * (b - a) + a

            def inv(y: torch.Tensor):
                return (y - a) / (b - a) * (hi_t - lo_t) + lo_t

        elif method == "standard":
            mu_t = torch.tensor(self._label_stats["mean"], dtype=torch.float32)
            sd_t = torch.tensor(self._label_stats["std"],  dtype=torch.float32)

            def fwd(y: torch.Tensor):
                return (y - mu_t) / sd_t

            def inv(y: torch.Tensor):
                return y * sd_t + mu_t
        else:
            raise RuntimeError("Scaler state corrupted.")

        return fwd, inv

    def _apply_label_scaler(self):
        fwd, inv = self._make_label_transforms()
        for ds in [self.train_ds, self.valid_ds, self.test_ds]:
            if ds is not None and ds.labels is not None:
                ds.set_label_transform(fwd, inv)

    def _save_label_scaler(self):
        os.makedirs(self.save_dir, exist_ok=True)
        st = self._label_stats
        if st["method"] == "minmax":
            np.savez(self.scaler_path, method="minmax",
                     min=st["min"], max=st["max"], a=st["a"], b=st["b"])
        else:
            np.savez(self.scaler_path, method="standard",
                     mean=st["mean"], std=st["std"])

    def _try_load_label_scaler(self) -> bool:
        try:
            z = np.load(self.scaler_path, allow_pickle=False)
        except Exception:
            return False

        method = str(z["method"])
        if method == "minmax":
            self._label_stats = {
                "method": "minmax",
                "min": z["min"], "max": z["max"],
                "a": float(z["a"]), "b": float(z["b"]),
            }
        elif method == "standard":
            self._label_stats = {
                "method": "standard",
                "mean": z["mean"], "std": z["std"],
            }
        else:
            return False

        self._scaler_fitted = True
        return True

