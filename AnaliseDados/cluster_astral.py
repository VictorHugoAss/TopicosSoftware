import os
import sys
import re
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, f1_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS, Birch, SpectralClustering, MeanShift, AffinityPropagation
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


DATASET_PATH = r"C:\Users\victo\OneDrive\Desktop\Faculdade\4º Semestre\Tópicos Software\Analise Dados\astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa.txt"
RESULTS_DIR = "results"
S_MAX = 2                
PCA_N = 300             
MAX_SEQS = 500 # Teste     
RANDOM_STATE = 42


def load_fasta_like(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    seqs = []
    header = None
    seq_lines = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    seqs.append((header, ''.join(seq_lines).upper()))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            seqs.append((header, ''.join(seq_lines).upper()))
    return seqs

def extract_class_from_header(header):
    tokens = header.split()

    for t in tokens:
        if re.match(r'^[a-z]\.\d+(\.\d+)*', t):
            return t
    return tokens[0] if tokens else header




AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {a:i for i,a in enumerate(AMINO_ACIDS)}

def all_dipeptides():
    return [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]

DIPEPS = all_dipeptides()

def build_feature_index(max_skip=2):
    features = []
    for s in range(max_skip+1):
        for dp in DIPEPS:
            features.append(f"{dp}|{s}")
    idx = {f:i for i,f in enumerate(features)}
    return features, idx

def sequence_to_binary_features(seq, idx_map, max_skip=2):
    L = len(seq)
    vec = np.zeros(len(idx_map), dtype=np.uint8)
    for s in range(max_skip+1):

        for i in range(L - (1 + s)):
            a = seq[i]
            b = seq[i+1+s]
            if a in AA_INDEX and b in AA_INDEX:
                key = f"{a}{b}|{s}"
                j = idx_map.get(key, None)
                if j is not None:
                    vec[j] = 1
    return vec

def build_binary_matrix(seqs, max_skip=2, max_sequences=None, verbose=True):
    features, idx_map = build_feature_index(max_skip=max_skip)
    n_features = len(features)
    n_sequences = len(seqs) if max_sequences is None else min(len(seqs), max_sequences)
    X = np.zeros((n_sequences, n_features), dtype=np.uint8)
    y = []
    headers = []
    skipped = 0
    for i, (header, seq) in enumerate(seqs[:n_sequences]):
        if len(seq) < 2:
            X[i,:] = 0
            skipped += 1
        else:
            X[i,:] = sequence_to_binary_features(seq, idx_map, max_skip=max_skip)
        y.append(extract_class_from_header(header))
        headers.append(header)
        if verbose and (i+1) % 500 == 0:
            print(f"  processed {i+1}/{n_sequences}")
    if verbose:
        print(f"Skipped (too short) sequences: {skipped}")
    return X, np.array(y), headers, features


def pca_transform(X, n_components=300):
    scaler = StandardScaler(with_mean=False)  
    Xs = scaler.fit_transform(X)
    n_comp = min(n_components, Xs.shape[1]-1, Xs.shape[0]-1)
    if n_comp < 1:
        raise ValueError("PCA não pode ter <1 componente: verifique dimensão dos dados")
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(Xs)
    return Xp, scaler, pca


def best_label_mapping(true_labels, pred_labels):
    labels_true, inv_true = np.unique(true_labels, return_inverse=True)
    labels_pred, inv_pred = np.unique(pred_labels, return_inverse=True)
    n_true = len(labels_true)
    n_pred = len(labels_pred)
    M = np.zeros((n_pred, n_true), dtype=np.int64)
    for p, t in zip(inv_pred, inv_true):
        M[p,t] += 1

    row_ind, col_ind = linear_sum_assignment(M.max() - M)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        mapping[labels_pred[r]] = labels_true[c]
    mapped = np.array([mapping.get(lb, -1) for lb in pred_labels], dtype=object)
    return mapped, mapping

def compute_external_f1(true_labels, pred_labels, average='macro'):
    mapped, mapping = best_label_mapping(true_labels, pred_labels)

    try:
        f1 = f1_score(true_labels, mapped, average=average, zero_division=0)
    except Exception:
        f1 = np.nan
    return f1

def compute_internal_metrics(X, labels):
    metrics = {'silhouette': np.nan, 'calinski_harabasz': np.nan, 'davies_bouldin': np.nan}
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1 or len(unique_labels) >= len(labels):
        return metrics
    try:
        metrics['silhouette'] = silhouette_score(X, labels)
    except Exception:
        metrics['silhouette'] = np.nan
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
    except Exception:
        metrics['calinski_harabasz'] = np.nan
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    except Exception:
        metrics['davies_bouldin'] = np.nan
    return metrics


def run_clustering_grid(Xp, y_true, out_dir=RESULTS_DIR, verbose=True):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    results = []
    grids = [
        ('KMeans', KMeans, {'n_clusters': list(range(2, 16)), 'random_state':[RANDOM_STATE]}),
        ('Agglomerative', AgglomerativeClustering, {'n_clusters': list(range(2, 16)), 'linkage':['ward','complete','average']}),
        ('DBSCAN', DBSCAN, {'eps':[0.5, 1.0, 1.5, 2.0], 'min_samples':[5,10,20]}),
        ('OPTICS', OPTICS, {'min_samples':[5,10,20], 'max_eps':[np.inf, 2.0, 5.0]}),
        ('Birch', Birch, {'n_clusters': list(range(2,16)), 'threshold':[0.5, 1.5, 2.5]}),
        ('Spectral', SpectralClustering, {'n_clusters': list(range(2,11)), 'assign_labels':['kmeans','discretize'], 'random_state':[RANDOM_STATE]}),
        ('MeanShift', MeanShift, {'bandwidth':[None]}),
        ('AffinityPropagation', AffinityPropagation, {'damping':[0.5, 0.7, 0.9]})
    ]

    for name, Cls, param_grid in grids:
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:

                model = Cls(**{k:v for k,v in params.items() if v is not None})

                if name in ['KMeans','Birch','Spectral','MeanShift','AffinityPropagation']:
                    labels = model.fit_predict(Xp)
                else:
                    model.fit(Xp)
                    if hasattr(model, 'labels_'):
                        labels = model.labels_
                    else:
                        labels = model.fit_predict(Xp)
                n_clusters = len(np.unique(labels[labels>=0]))
                metrics_internal = compute_internal_metrics(Xp, labels)
                external_f1 = compute_external_f1(y_true, labels, average='macro')
                res = {
                    'algorithm': name,
                    'params': str(params),
                    'n_clusters': int(n_clusters),
                    'labels_noise': int(np.sum(labels==-1)) if np.any(labels==-1) else 0,
                    'external_f1_macro': float(external_f1) if not np.isnan(external_f1) else np.nan,
                    'silhouette': float(metrics_internal['silhouette']) if not np.isnan(metrics_internal['silhouette']) else np.nan,
                    'calinski_harabasz': float(metrics_internal['calinski_harabasz']) if not np.isnan(metrics_internal['calinski_harabasz']) else np.nan,
                    'davies_bouldin': float(metrics_internal['davies_bouldin']) if not np.isnan(metrics_internal['davies_bouldin']) else np.nan
                }
                results.append(res)
                if verbose:
                    print(f"[{name}] {params} -> clusters {n_clusters}, F1={external_f1:.4f}, Sil={metrics_internal['silhouette']}")
            except Exception as e:
                if verbose:
                    print(f"Erro em {name} {params}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(out_dir, "clustering_results.csv"), index=False)
    return df


def select_best_by_correlation(df, internal_metrics=['silhouette','calinski_harabasz','davies_bouldin']):
    corr = {}
    for m in internal_metrics:
        valid = df[[m,'external_f1_macro']].dropna()
        if len(valid) > 2:
            corr_val, p = spearmanr(valid[m], valid['external_f1_macro'])
        else:
            corr_val = np.nan
        corr[m] = corr_val

    best_metric = max(corr.items(), key=lambda kv: (0 if np.isnan(kv[1]) else abs(kv[1])))[0]

    if best_metric == 'davies_bouldin':
        best_row = df.loc[df[best_metric].idxmin()]
    else:
        best_row = df.loc[df[best_metric].idxmax()]
    return corr, best_metric, best_row


def make_plots(df, out_dir=RESULTS_DIR):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for m in ['silhouette','calinski_harabasz','davies_bouldin']:
        plt.figure(figsize=(6,4))
        plt.scatter(df[m], df['external_f1_macro'])
        plt.xlabel(m)
        plt.ylabel('external_f1_macro')
        plt.title(f'F1 vs {m}')
        plt.grid(True)
        plt.tight_layout()
        fn = os.path.join(out_dir, f"fig_f1_vs_{m}.png")
        plt.savefig(fn)
        plt.close()


def main():
    print("Carregando sequências do dataset (ASTRAL/SCOP)...")
    seqs = load_fasta_like(DATASET_PATH)
    print(f"Total sequências lidas: {len(seqs)}")
    if MAX_SEQS is not None:
        print(f"Usando no máximo {MAX_SEQS} sequências para rapidez.")
    print("Construindo matriz binária de features (dipeptídeos + skip)...")
    X, y, headers, features = build_binary_matrix(seqs, max_skip=S_MAX, max_sequences=MAX_SEQS, verbose=True)
    print("Matriz X shape:", X.shape)
    print("Executando PCA...")
    Xp, scaler, pca = pca_transform(X, n_components=PCA_N)
    print("PCA result shape:", Xp.shape)
    print("Rodando grid de clustering (vários algoritmos) — isso pode demorar...")
    df_results = run_clustering_grid(Xp, y, out_dir=RESULTS_DIR, verbose=True)
    print("Selecionando melhor configuração por correlação entre métricas internas e F1...")
    corr, best_metric, best_row = select_best_by_correlation(df_results)
    print("Correlações (Spearman) entre métricas internas e F1:", corr)
    print("Métrica interna escolhida:", best_metric)
    print("Melhor configuração segundo esse critério:\n", best_row.to_dict())
    joblib.dump({'scaler':scaler, 'pca':pca, 'features':features}, os.path.join(RESULTS_DIR, 'pipeline_objects.joblib'))
    df_results.to_csv(os.path.join(RESULTS_DIR, "clustering_results.csv"), index=False)
    print("Gerando gráficos simples...")
    make_plots(df_results, out_dir=RESULTS_DIR)
    print(f"Terminou. Resultados salvos em '{RESULTS_DIR}/' (clustering_results.csv, fig_*.png, pipeline_objects.joblib).")

if __name__ == "__main__":
    main()
