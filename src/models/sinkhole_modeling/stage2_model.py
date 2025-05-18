import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from src.models.sinkhole_modeling.config import log, LGB_PARAMS_BASE, GRAPHSAGE_PARAMS
from src.models.sinkhole_modeling.graphsage import GraphSAGE, UncertaintyMaskedGraphSAGE
from sklearn.ensemble import IsolationForest

def train_stage2_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    stage1_scores_train: np.ndarray,
    stage1_scores_val: np.ndarray,
    coords_train: Optional[np.ndarray],
    coords_val: Optional[np.ndarray],
    exclude_features: List[str] = None,
    model_type: str = 'lightgbm',  # 'lightgbm', 'gnn', 'um_gnn'
    feature_fraction: float = 0.6,
    min_positives: int = 5,
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6
) -> Tuple[Optional[object], Dict[str, float], bool]:
    """
    Train Stage 2 model: supervised LightGBM, GNN/UM-GNN, or unsupervised IsolationForest.
    Returns (model, metrics, is_unsupervised).
    """
    # 1) 컬럼 필터링: 객체형, geometry, grid_id, observed sinkhole 특성 제외
    forbidden = set((exclude_features or []) + ['geometry', 'grid_id'])
    features = [
        c for c in X_train.columns
        if c not in forbidden and not pd.api.types.is_object_dtype(X_train[c])
    ]

    # 2) Positive label 수 체크
    supervised_ok = (y_train.sum() >= min_positives) and (y_val.sum() >= min_positives)

    # -------------------
    # 3) GNN/UM-GNN 분기
    # -------------------
    if model_type in ['gnn', 'um_gnn']:
        log(f"Training {model_type} model with {len(features)} features", level=1)
        assert coords_train is not None and coords_val is not None, "coords required for GNN"
        # PyG 데이터 객체 생성
        def build_data(Xf, coords, y):
            radius = GRAPHSAGE_PARAMS.get('radius', 1000)
            ei = radius_graph(torch.tensor(coords, dtype=torch.float),
                              r=GRAPHSAGE_PARAMS['radius'], loop=True)
            return Data(
                x=torch.tensor(Xf[features].values, dtype=torch.float),
                y=torch.tensor(y.values, dtype=torch.float),
                edge_index=ei
            )

        data_tr = build_data(X_train, coords_train, y_train)
        data_val = build_data(X_val,   coords_val,   y_val)

        Model = GraphSAGE if model_type == 'gnn' else UncertaintyMaskedGraphSAGE
        kwargs = {}
        if model_type == 'um_gnn':
            kwargs = {
                'uncertainty_low': uncertainty_low,
                'uncertainty_high': uncertainty_high
            }

        model = Model(
            in_channels=data_tr.num_features,
            hidden_channels=GRAPHSAGE_PARAMS['hidden_channels'],
            num_layers=GRAPHSAGE_PARAMS['num_layers'],
            dropout=GRAPHSAGE_PARAMS['dropout'],
            jk=GRAPHSAGE_PARAMS['jk'],
            **kwargs
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        data_tr, data_val = data_tr.to(device), data_val.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=GRAPHSAGE_PARAMS['lr'])

        best_auc, patience, best_state = 0.0, 0, None
        train_mask = torch.ones(data_tr.num_nodes, dtype=torch.bool, device=device)

        for epoch in range(GRAPHSAGE_PARAMS['epochs']):
            model.train(); optimizer.zero_grad()


            s1_in = torch.tensor(stage1_scores_train, dtype=torch.float, device=device)
            out = model(data_tr.x, data_tr.edge_index, s1_in if model_type=='um_gnn' else None)
            loss = F.binary_cross_entropy(out[train_mask], data_tr.y[train_mask])
            loss.backward(); optimizer.step()


            model.eval()
            with torch.no_grad():
                s1_val_in = torch.tensor(stage1_scores_val, dtype=torch.float, device=device)
                val_out = model(data_val.x, data_val.edge_index, s1_val_in if model_type=='um_gnn' else None)
                preds = val_out.cpu().numpy()
                auc_val = roc_auc_score(data_val.y.cpu().numpy(), preds)

            if epoch % 10 == 0:  # 10 에포크마다 로그 출력
                log(f"Epoch {epoch}: loss={loss.item():.4f}, val_auc={auc_val:.4f}", level=2)

            if auc_val > best_auc:
                best_auc, best_state, patience = auc_val, model.state_dict(), 0
            else:
                patience += 1
                if patience >= GRAPHSAGE_PARAMS['patience']:
                    break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            s1_val_in = torch.tensor(stage1_scores_val, dtype=torch.float, device=device)
            final_out = model(data_val.x, data_val.edge_index, s1_val_in if model_type=='um_gnn' else None)
            final_preds = final_out.cpu().numpy()

        pr, rc, _ = precision_recall_curve(data_val.y.cpu().numpy(), final_preds)
        return model, {
            'auc': roc_auc_score(data_val.y.cpu().numpy(), final_preds),
            'pr_auc': auc(rc, pr)
        }, False

    # --------------------------------
    # 4) LightGBM supervised fallback
    # --------------------------------
    if supervised_ok and model_type == 'lightgbm':
        import lightgbm as lgb
        params = LGB_PARAMS_BASE.copy()
        neg, pos = (y_train==0).sum(), y_train.sum()
        params.update({
            'scale_pos_weight': neg/pos,
            'feature_fraction': feature_fraction
        })
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train[features], y_train,
            eval_set=[(X_val[features], y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        val_preds = model.predict_proba(X_val[features])[:,1]
        pr, rc, _ = precision_recall_curve(y_val, val_preds)
        return model, {
            'auc': roc_auc_score(y_val, val_preds),
            'pr_auc': auc(rc, pr)
        }, False

    # ------------------------------------
    # 5) Unsupervised IsolationForest fallback
    # ------------------------------------
    iso = IsolationForest(contamination=0.05, random_state=42)
    # DataFrame 전체로 fit
    iso.fit(pd.concat([X_train[features], X_val[features]], axis=0))
    raw = -iso.decision_function(X_val[features])
    norm = (raw - raw.min())/(raw.max()-raw.min()+1e-8)
    pr, rc, _ = precision_recall_curve(y_val, norm)
    return iso, {
        'auc': roc_auc_score(y_val, norm),
        'pr_auc': auc(rc, pr)
    }, True