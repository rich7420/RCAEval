import argparse
import glob
import json
import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from multiprocessing import Pool
from os.path import abspath, basename, dirname, exists, join

# turn off all warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from RCAEval.benchmark.evaluation import Evaluator
from RCAEval.classes.graph import Node

from RCAEval.io.time_series import drop_constant, drop_time, preprocess
from RCAEval.utility import (
    dump_json,
    is_py38,
    is_py310,
    load_json,
    download_online_boutique_dataset,
    download_sock_shop_1_dataset,
    download_sock_shop_2_dataset,
    download_train_ticket_dataset,
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset,
    download_syn_circa_dataset,
    download_syn_rcd_dataset,
    download_syn_causil_dataset,
    download_multi_source_sample,
)


if is_py310():
    from RCAEval.e2e import (
        baro,
        causalrca,
        circa,
        cloudranger,
        cmlp_pagerank,
        dummy,
        e_diagnosis,
        easyrca,
        fci_pagerank,
        fci_randomwalk,
        ges_pagerank,
        granger_pagerank,
        granger_randomwalk,
        lingam_pagerank,
        lingam_randomwalk,
        micro_diag,
        microcause,
        microrank,
        mscred,
        nsigma,
        ntlr_pagerank,
        ntlr_randomwalk,
        pc_pagerank,
        pc_randomwalk,
        rcd,
        run,
        tracerca,
    )
    # Add GNN+KAN method
    try:
        from RCAEval.e2e.gnnkan import gnn_kan_rca
        print("✅ GNN+KAN method imported successfully")
    except ImportError as e:
        print(f"⚠️ GNN+KAN import failed: {e}")
        gnn_kan_rca = None
    
    # Add pure GNN method
    try:
        from RCAEval.e2e.gnn import gnn_rca
        print("✅ Pure GNN method imported successfully")
    except ImportError as e:
        print(f"⚠️ Pure GNN import failed: {e}")
        gnn_rca = None
    
    # Add GAT method (fair baseline for GNN-KAN)
    try:
        from RCAEval.e2e.gat import gat_rca
        print("✅ GAT method imported successfully")
    except ImportError as e:
        print(f"⚠️ GAT import failed: {e}")
        gat_rca = None
    
    # Add GATv2 method (improved GAT)
    try:
        from RCAEval.e2e.gatv2 import gatv2_rca
        print("✅ GATv2 method imported successfully")
    except ImportError as e:
        print(f"⚠️ GATv2 import failed: {e}")
        gatv2_rca = None
    
    # Add Graph Transformer method (modern GNN)
    try:
        from RCAEval.e2e.graph_transformer import graph_transformer_rca
        print("✅ Graph Transformer method imported successfully")
    except ImportError as e:
        print(f"⚠️ Graph Transformer import failed: {e}")
        graph_transformer_rca = None

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
    gnn_kan_rca = None  # GNN+KAN requires Python 3.10+
    gnn_rca = None  # Pure GNN also requires Python 3.9+
    gat_rca = None  # GAT also requires Python 3.9+
    gatv2_rca = None  # GATv2 also requires Python 3.9+
    graph_transformer_rca = None  # Graph Transformer also requires Python 3.9+
elif sys.version_info[:2] == (3, 9) or is_py310():
    # Support Python 3.9 and 3.10+ by using the same imports
    # Python 3.9 can use the same methods as Python 3.10
    from RCAEval.e2e import (
        baro,
        causalrca,
        circa,
        cloudranger,
        cmlp_pagerank,
        dummy,
        e_diagnosis,
        easyrca,
        fci_pagerank,
        fci_randomwalk,
        ges_pagerank,
        granger_pagerank,
        granger_randomwalk,
        lingam_pagerank,
        lingam_randomwalk,
        micro_diag,
        microcause,
        microrank,
        mscred,
        nsigma,
        ntlr_pagerank,
        ntlr_randomwalk,
        pc_pagerank,
        pc_randomwalk,
        rcd,
        run,
        tracerca,
    )
    # Add GNN+KAN method
    try:
        from RCAEval.e2e.gnnkan import gnn_kan_rca
        print("✅ GNN+KAN method imported successfully")
    except ImportError as e:
        print(f"⚠️ GNN+KAN import failed: {e}")
        gnn_kan_rca = None
    
    # Add pure GNN method
    try:
        from RCAEval.e2e.gnn import gnn_rca
        print("✅ Pure GNN method imported successfully")
    except ImportError as e:
        print(f"⚠️ Pure GNN import failed: {e}")
        gnn_rca = None
    
    # Add GAT method (fair baseline for GNN-KAN)
    try:
        from RCAEval.e2e.gat import gat_rca
        print("✅ GAT method imported successfully")
    except ImportError as e:
        print(f"⚠️ GAT import failed: {e}")
        gat_rca = None
    
    # Add GATv2 method (improved GAT)
    try:
        from RCAEval.e2e.gatv2 import gatv2_rca
        print("✅ GATv2 method imported successfully")
    except ImportError as e:
        print(f"⚠️ GATv2 import failed: {e}")
        gatv2_rca = None
    
    # Add Graph Transformer method (modern GNN)
    try:
        from RCAEval.e2e.graph_transformer import graph_transformer_rca
        print("✅ Graph Transformer method imported successfully")
    except ImportError as e:
        print(f"⚠️ Graph Transformer import failed: {e}")
        graph_transformer_rca = None
else:
    print("Please use Python 3.8, 3.9, or 3.10+")
    exit(1)

try:
    import torch
    # 不再全局禁用 GPU，讓各個方法自己決定是否使用 GPU
    # 對於需要禁用 GPU 的方法（如 causalrca），可以在方法內部設置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 已移除：允許使用 GPU
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
        "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt",
        "syn_circa", "syn_rcd", "syn_causil", "multi-source"
    ])
    parser.add_argument("--length", type=int, default=20, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run on all data")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate for gnn_kan_rca")
    parser.add_argument("--basis_function", type=str, default="chebyshev", 
                       choices=["chebyshev", "b_spline", "fourier", "pqc", "pqc_gpu"], 
                       help="Choose basis function for gnn_kan_rca")
    parser.add_argument(
        "--feature_method",
        type=str,
        default=None,
        choices=[
            "enhanced_ica", "ica", "simplified", "kpca", "rca_aware", "multimodal_fusion", "auto"
        ],
        help="Choose feature extraction method for gnn_kan_rca (e.g., rca_aware)"
    )
    args = parser.parse_args()

    # Check if method is available (including GNN+KAN and pure GNN)
    available_methods = []
    if is_py310() or sys.version_info[:2] == (3, 9):
        available_methods = [
            "baro", "causalrca", "circa", "cloudranger", "cmlp_pagerank", "dummy",
            "e_diagnosis", "easyrca", "fci_pagerank", "fci_randomwalk", "ges_pagerank",
            "granger_pagerank", "granger_randomwalk", "lingam_pagerank", "lingam_randomwalk",
            "micro_diag", "microcause", "microrank", "mscred", "nsigma", "ntlr_pagerank",
            "ntlr_randomwalk", "pc_pagerank", "pc_randomwalk", "rcd", "run", "tracerca"
        ]
        if gnn_kan_rca is not None:
            available_methods.append("gnn_kan_rca")
        if gnn_rca is not None:
            available_methods.append("gnn_rca")
        if gat_rca is not None:
            available_methods.append("gat_rca")
        if gatv2_rca is not None:
            available_methods.append("gatv2_rca")
        if graph_transformer_rca is not None:
            available_methods.append("graph_transformer_rca")
    elif is_py38():
        available_methods = ["dummy", "e_diagnosis", "ht", "rcd", "mmrcd"]
    
    if args.method not in available_methods:
        raise ValueError(f"{args.method=} not defined. Available methods: {available_methods}")

    return args


args = parse_args()

# download dataset (skip if already exists to avoid read-only errors)
DATASET_MAP = {
    "online-boutique": "data/online-boutique",
    "sock-shop-1": "data/sock-shop-1",
    "sock-shop-2": "data/sock-shop-2",
    "train-ticket": "data/train-ticket",
    "re1-ob": "data/online-boutique",
    "re1-ss": "data/sock-shop-2",
    "re1-tt": "data/train-ticket",
    "re2-ob": "data/RE2/RE2-OB",
    "re2-ss": "data/RE2/RE2-SS",
    "re2-tt": "data/RE2/RE2-TT",
    "re3-ob": "data/RE3/RE3-OB",
    "re3-ss": "data/RE3/RE3-SS",
    "re3-tt": "data/RE3/RE3-TT",
    "syn_circa": "data/syn_circa",
    "syn_rcd": "data/syn_rcd",
    "syn_causil": "data/syn_causil",
    "multi-source": "data/multi-source-data"
}
desired_dataset_path = DATASET_MAP.get(args.dataset)
if desired_dataset_path is None:
    raise Exception(f"{args.dataset} is not defined!")

if not os.path.exists(desired_dataset_path):
    if "online-boutique" in args.dataset or "re1-ob" in args.dataset:
        download_online_boutique_dataset()
    elif "sock-shop-1" in args.dataset:
        download_sock_shop_1_dataset()
    elif "sock-shop-2" in args.dataset or "re1-ss" in args.dataset:
        download_sock_shop_2_dataset()
    elif "train-ticket" in args.dataset or "re1-tt" in args.dataset:
        download_train_ticket_dataset()
    elif "re2" in args.dataset:
        download_re2_dataset()
    elif "re3" in args.dataset:
        download_re3_dataset()
    elif "syn_circa" in args.dataset:
        download_syn_circa_dataset()
    elif "syn_rcd" in args.dataset:
        download_syn_rcd_dataset()
    elif "syn_causil" in args.dataset:
        download_syn_causil_dataset()
    elif "multi-source" in args.dataset:
        download_multi_source_sample()
    else:
        raise Exception(f"{args.dataset} is not defined!")

dataset = DATASET_MAP[args.dataset]


# prepare input paths
data_paths = list(glob.glob(os.path.join(dataset, "**/data.csv"), recursive=True))
if not data_paths: 
    data_paths = list(glob.glob(os.path.join(dataset, "**/simple_metrics.csv"), recursive=True))
# new_data_paths = []
# for p in data_paths: 
#     if os.path.exists(p.replace("data.csv", "simple_data.csv")):
#         new_data_paths.append(p.replace("data.csv", "simple_data.csv"))
#     elif os.path.exists(p.replace("data.csv", "simple_metrics.csv")):
#         new_data_paths.append(p.replace("data.csv", "simple_metrics.csv"))
#     else:
#         new_data_paths.append(p)
# data_paths = new_data_paths
if args.test is True:
    data_paths = data_paths[:2]


# prepare output paths
from tempfile import TemporaryDirectory
# output_path = TemporaryDirectory().name
output_path = "output"
report_path = join(output_path, f"report.xlsx")
result_path = join(output_path, "results")
os.makedirs(result_path, exist_ok=True)


def process(data_path):
    run_args = argparse.Namespace()
    run_args.root_path = os.getcwd()
    run_args.data_path = data_path
    
    # convert length from minutes to seconds
    if args.length is None:
        args.length = 10
    data_length = args.length * 60 // 2

    data_dir = dirname(data_path)

    # 處理可能包含多個下劃線的目錄名稱
    parent_dir_name = basename(dirname(dirname(data_path)))
    parts = parent_dir_name.split("_")
    if len(parts) >= 2:
        service = parts[0]
        metric = "_".join(parts[1:])  # 將剩餘部分重新組合為 metric
    else:
        # 如果沒有下劃線，嘗試其他分隔符或使用整個名稱
        service = parent_dir_name
        metric = "unknown"
    case = basename(dirname(data_path))

    rp = join(result_path, f"{service}_{metric}_{case}.json")

    # == Load and Preprocess data ==
    data = pd.read_csv(data_path)
    
    # remove lat-50, only selecte lat-90 
    data = data.loc[:, ~data.columns.str.endswith("_latency-50")]
    
    if "mm-tt" in data_path:
        time_col = data["time"]
        data = data.loc[:, data.columns.str.startswith("ts-")]
        data["time"] = time_col
        
    # handle inf
    data = data.replace([np.inf, -np.inf], np.nan)

    # handle na
    data = data.fillna(method="ffill")
    data = data.fillna(0)

    # 獲取注入時間 - 支持多種數據集格式
    inject_time = None
    inject_time_file = join(data_dir, "inject_time.txt")
    
    # 1. 嘗試讀取 inject_time.txt（標準數據集格式）
    if os.path.exists(inject_time_file):
        try:
            with open(inject_time_file) as f:
                inject_time = int(f.readlines()[0].strip()) + args.tdelta
        except Exception as e:
            print(f"⚠️ 讀取 inject_time.txt 失敗: {e}")
    
    # 2. 如果沒有 inject_time.txt，嘗試從 info.json 讀取（合成數據集格式）
    if inject_time is None:
        info_file = join(data_dir, "info.json")
        if os.path.exists(info_file):
            try:
                import json
                with open(info_file) as f:
                    info = json.load(f)
                if "length_normal" in info:
                    inject_time = info["length_normal"]
                    print(f"📅 從 info.json 讀取注入時間: {inject_time}")
            except Exception as e:
                print(f"⚠️ 讀取 info.json 失敗: {e}")
    
    # 3. 如果都沒有，使用數據本身推斷
    if inject_time is None:
        if "time" in data.columns:
            inject_time = int(data["time"].median())
            print(f"📅 使用時間列中位數作為注入時間: {inject_time}")
        else:
            inject_time = len(data) // 2
            print(f"📅 使用數據中點作為注入時間: {inject_time}")
    
    # 確保數據有 time 列（合成數據集可能沒有）
    if "time" not in data.columns:
        data["time"] = data.index
    
    # for metrics, minutes -> seconds // 2
    normal_df = data[data["time"] < inject_time].tail(args.length * 60 // 2)
    anomal_df = data[data["time"] >= inject_time].head(args.length * 60 // 2)

    data = pd.concat([normal_df, anomal_df], ignore_index=True)

    # num column, exclude time
    num_node = len(data.columns) - 1

    # rename latency
    data = data.rename(
        columns={
            c: c.replace("_latency-90", "_latency")
            for c in data.columns
            if c.endswith("_latency-90")
        }
    )
    
    # == Get SLI ===
    sli = None
    if "my-sock-shop" in data_path or "fse-ss" in data_path:
        sli = "front-end_cpu"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "sock-shop" in data_path:
        sli = "front-end_cpu"
        if f"{service}_lat_90" in data:
            sli = f"{service}_lat_90"
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path or "RE3-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path or "RE3-OB" in data_path or "RE3-SS" in data_path:
        sli = "frontend_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        elif "frontend_1" in data:
            sli = "frontend_1"
    elif "syn_circa" in data_path or "syn_rcd" in data_path or "syn_causil" in data_path:
        # 合成數據集：嘗試從數據中識別 SLI，如果沒有則設為 None
        # 合成數據集通常沒有明確的 SLI，許多方法可以處理 sli=None
        sli_candidates = [col for col in data.columns if any(keyword in str(col).lower() 
                          for keyword in ['latency', 'error', 'lat', 'delay', 'response'])]
        if sli_candidates:
            sli = sli_candidates[0]
            print(f"📊 合成數據集：使用 {sli} 作為 SLI")
        else:
            sli = None
            print(f"📊 合成數據集：未找到明確的 SLI，將使用 None（某些方法可能不需要 SLI）")
    elif "multi-source" in data_path:
        # 多源數據集：嘗試識別 SLI
        sli_candidates = [col for col in data.columns if any(keyword in str(col).lower() 
                          for keyword in ['latency', 'error', 'lat', 'delay', 'response'])]
        if sli_candidates:
            sli = sli_candidates[0]
            print(f"📊 多源數據集：使用 {sli} 作為 SLI")
        else:
            sli = None
            print(f"📊 多源數據集：未找到明確的 SLI，將使用 None")
    else:
        raise ValueError(f"SLI not implemented for dataset: {args.dataset}")

    # == PROCESS ==
    func = globals()[args.method]

    try:
        st = datetime.now()
        
        # Handle GNN+KAN method with specific parameters
        if args.method == "gnn_kan_rca":
            # 提取故障類型信息
            fault_type = metric  # 從文件名提取: cpu, mem, disk, socket, delay, loss
            
            out = func(
                data,
                inject_time,
                dataset=args.dataset,
                config_type="simplified",
                feature_method=(args.feature_method if args.feature_method is not None else "enhanced_ica"),
                use_optimized_input=True,
                sparsity_lambda=2e-3,           # 大幅增加稀疏性權重
                # 🎯 高精度訓練參數
                learning_rate=(args.learning_rate if args.learning_rate is not None else 1.07655e-7),           # 優化學習率
                num_epochs=400,                 # 大幅增加訓練輪數
                kan_grid_size=20,               # 大幅增加KAN網格
                hidden_dim=128,                 # 增加隱藏維度
                target_feature_dim=128,         # 增加特徵維度
                similarity_threshold=0.265,       # 大幅降低相似性閾值
                max_edges_per_node=15,          # 大幅增加邊數
                gradient_clipping=0.6,          # 更嚴格的梯度裁剪
                # 🎯 新增監督學習參數
                fault_type=fault_type,          # 故障類型信息（關鍵！）
                enhanced_contrast=True,         # 啟用增強對比學習
                fault_type_aware=True,          # 啟用故障類型感知
                adaptive_learning=True,         # 啟用自適應學習
                multi_scale_features=True,      # 啟用多尺度特徵
                temporal_attention=True,        # 啟用時序注意力
                # 🎯 基函數選擇參數
                basis_function=args.basis_function,  # 基函數選擇
                sli=sli,
                verbose=True
            )
        # Handle pure GNN method with specific parameters
        elif args.method == "gnn_rca":
            out = func(
                data,
                inject_time,
                dataset=args.dataset,
                config_type="simplified",
                feature_method=(args.feature_method if args.feature_method is not None else "enhanced_ica"),
                use_optimized_input=True,
                learning_rate=1e-3,
                num_epochs=100,
                hidden_dim=64,
                target_feature_dim=64,
                similarity_threshold=0.3,
                max_edges_per_node=5,
                sli=sli,
                verbose=False
            )
        # Handle GAT method with specific parameters (fair baseline for GNN-KAN)
        elif args.method == "gat_rca":
            out = func(
                data,
                inject_time,
                dataset=args.dataset,
                config_type="simplified",
                feature_method=(args.feature_method if args.feature_method is not None else "enhanced_ica"),
                use_optimized_input=True,
                # 與 GNN-KAN 完全相同的參數，確保公平比較
                learning_rate=2e-4,
                num_epochs=400,
                num_gnn_layers=4,
                hidden_dims=[128, 96, 64],
                output_dim=96,
                input_dim=128,
                target_feature_dim=128,
                similarity_threshold=0.2,
                max_edges_per_node=15,
                dropout=0.1,
                weight_decay=5e-6,
                gradient_clip_norm=0.5,
                patience=60,
                sli=sli,
                verbose=False
            )
        # Handle GATv2 method with specific parameters (improved GAT baseline)
        elif args.method == "gatv2_rca":
            out = func(
                data,
                inject_time,
                dataset=args.dataset,
                config_type="simplified",
                feature_method=(args.feature_method if args.feature_method is not None else "enhanced_ica"),
                use_optimized_input=True,
                # 與 GNN-KAN 完全相同的參數，確保公平比較
                learning_rate=2e-4,
                num_epochs=400,
                num_gnn_layers=4,
                hidden_dims=[128, 96, 64],
                output_dim=96,
                input_dim=128,
                target_feature_dim=128,
                similarity_threshold=0.2,
                max_edges_per_node=15,
                dropout=0.1,
                weight_decay=5e-6,
                gradient_clip_norm=0.5,
                patience=60,
                sli=sli,
                verbose=False
            )
        # Handle Graph Transformer method with specific parameters (modern GNN baseline)
        elif args.method == "graph_transformer_rca":
            out = func(
                data,
                inject_time,
                dataset=args.dataset,
                config_type="simplified",
                feature_method=(args.feature_method if args.feature_method is not None else "enhanced_ica"),
                use_optimized_input=True,
                # 與 GNN-KAN 完全相同的參數，確保公平比較
                learning_rate=2e-4,
                num_epochs=400,
                num_gnn_layers=4,
                hidden_dims=[128, 96, 64],
                output_dim=96,
                input_dim=128,
                target_feature_dim=128,
                similarity_threshold=0.2,
                max_edges_per_node=15,
                dropout=0.1,
                weight_decay=5e-6,
                gradient_clip_norm=0.5,
                patience=60,
                sli=sli,
                verbose=False
            )
        else:
            # Standard method execution
            out = func(
                data,
                inject_time,
                dataset=args.dataset,
                anomalies=None,
                dk_select_useful=False,
                sli=sli,
                verbose=False,
                n_iter=num_node,
                args=run_args,
            )
        
        root_causes = out.get("ranks")
        # print("==============")
        # print(f"{data_path=}")
        # print(root_causes[:5])
        dump_json(filename=rp, data={0: root_causes})
    except Exception as e:
        raise e
        print(f"{args.method=} failed on {data_path=}")
        print(e)
        rp = join(result_path, f"{service}_{metric}_{case}_failed.json")
        with open(rp, "w") as f:
            json.dump({"error": str(e)}, f)


start_time = datetime.now()

for data_path in tqdm(sorted(data_paths)):
    process(data_path)

end_time = datetime.now()
time_taken = end_time - start_time
avg_speed = round(time_taken.total_seconds() / len(data_paths), 2)


# ======== EVALUTION ===========
# 🔧 修復：只評估當前運行生成的文件，避免緩存問題
if args.test:
    # 測試模式：只讀取當前方法生成的文件
    # 注意：結果文件名格式是 {service}_{metric}_{case}.json，不是 {method}_*.json
    # 所以我們需要讀取所有文件，但只評估當前運行的數據路徑
    rps = []
    for data_path in data_paths:
        # 處理可能包含多個下劃線的目錄名稱
        parent_dir_name = basename(dirname(dirname(data_path)))
        parts = parent_dir_name.split("_")
        if len(parts) >= 2:
            service = parts[0]
            metric = "_".join(parts[1:])  # 將剩餘部分重新組合為 metric
        else:
            # 如果沒有下劃線，嘗試其他分隔符或使用整個名稱
            service = parent_dir_name
            metric = "unknown"
        case = basename(dirname(data_path))
        rp = join(result_path, f"{service}_{metric}_{case}.json")
        if exists(rp):
            rps.append(rp)
else:
    # 正常模式：讀取所有文件
    rps = glob.glob(join(result_path, "*.json"))

services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

# 故障類型映射：將 f1-f5 映射到標準故障類型
# 根據常見的故障類型順序：cpu, mem, disk, socket, delay, loss
fault_type_mapping = {
    'f1': 'cpu',
    'f2': 'mem', 
    'f3': 'disk',
    'f4': 'socket',
    'f5': 'delay',
    'f6': 'loss',
    # 標準故障類型保持不變
    'cpu': 'cpu',
    'mem': 'mem',
    'disk': 'disk',
    'socket': 'socket',
    'delay': 'delay',
    'loss': 'loss',
}

# 將故障類型映射到標準格式
faults_mapped = [fault_type_mapping.get(f, f) for f in faults]
# 去重並保持順序
faults = sorted(list(set(faults_mapped)))

eval_data = {
    "service-fault": [],
    "top_1_service": [],
    "top_3_service": [],
    "top_5_service": [],
    "avg@5_service": [],
    "top_1_metric": [],
    "top_3_metric": [],
    "top_5_metric": [],
    "avg@5_metric": [],
}

s_evaluator_all = Evaluator()
f_evaluator_all = Evaluator()
s_evaluator_cpu = Evaluator()
f_evaluator_cpu = Evaluator()
s_evaluator_mem = Evaluator()
f_evaluator_mem = Evaluator()
s_evaluator_lat = Evaluator()
f_evaluator_lat = Evaluator()
s_evaluator_loss = Evaluator()
f_evaluator_loss = Evaluator()
s_evaluator_io = Evaluator()
f_evaluator_io = Evaluator()
s_evaluator_socket = Evaluator()
f_evaluator_socket = Evaluator()

for service in services:
    for fault in faults:
        s_evaluator = Evaluator()
        f_evaluator = Evaluator()

        for rp in rps:
            s, m = basename(rp).split("_")[:2]
            # 映射故障類型
            m_mapped = fault_type_mapping.get(m, m)
            if s != service or m_mapped != fault:
                continue  # ignore

            data = load_json(rp)
            if "error" in data:
                continue  # ignore

            for i, ranks in data.items():
                # 過濾掉 IP 地址格式的字符串 (如 '192-168-29-237-9100')
                s_ranks = []
                for x in ranks:
                    service_name = x.split("_")[0].replace("-db", "")
                    # 檢查是否為 IP 地址格式 (如 192-168-xx-xx-xxxx)
                    is_ip_format = (
                        (service_name.startswith("192-168-") and service_name.count("-") >= 4) or
                        # 檢查是否符合 IP 地址的一般模式 (數字-數字-數字-數字-端口)
                        (service_name.count("-") >= 4 and all(part.isdigit() for part in service_name.split("-")))
                    )
                    if not is_ip_format:
                        s_ranks.append(Node(service_name, "unknown"))
                # remove duplication
                old_s_ranks = s_ranks.copy()
                if old_s_ranks:  # 確保列表不為空
                    s_ranks = [old_s_ranks[0]] + [
                        old_s_ranks[i]
                        for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                else:
                    s_ranks = []

                # Handle different output formats from different methods
                f_ranks = []
                for x in ranks:
                    service_name = x.split("_")[0] if "_" in x else x
                    
                    # 檢查是否為 IP 地址格式 (如 192-168-xx-xx-xxxx)
                    is_ip_format = (
                        (service_name.startswith("192-168-") and service_name.count("-") >= 4) or
                        # 檢查是否符合 IP 地址的一般模式 (數字-數字-數字-數字-端口)
                        (service_name.count("-") >= 4 and all(part.isdigit() for part in service_name.split("-")))
                    )
                    if is_ip_format:
                        continue  # 跳過 IP 地址格式的字符串
                        
                    if "_" in x and len(x.split("_")) >= 2:
                        # Standard format: service_metric
                        f_ranks.append(Node(service_name, x.split("_")[1]))
                    else:
                        # GNN+KAN format: just service name, use fault type as metric
                        f_ranks.append(Node(service_name, fault))

                s_evaluator.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                f_evaluator.add_case(ranks=f_ranks, answer=Node(service, fault))

                if fault == "cpu":
                    s_evaluator_cpu.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_cpu.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "mem":
                    s_evaluator_mem.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_mem.add_case(ranks=f_ranks, answer=Node(service, fault))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, fault))

                elif fault == "delay":
                    s_evaluator_lat.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_lat.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "loss":
                    s_evaluator_loss.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_loss.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "latency"))

                elif fault == "disk":
                    s_evaluator_io.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_io.add_case(ranks=f_ranks, answer=Node(service, "diskio"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "diskio"))
                elif fault == "socket":
                    s_evaluator_socket.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_socket.add_case(ranks=f_ranks, answer=Node(service, "socket"))

                    s_evaluator_all.add_case(ranks=s_ranks, answer=Node(service, "unknown"))
                    f_evaluator_all.add_case(ranks=f_ranks, answer=Node(service, "socket"))


        eval_data["service-fault"].append(f"{service}_{fault}")
        eval_data["top_1_service"].append(s_evaluator.accuracy(1))
        eval_data["top_3_service"].append(s_evaluator.accuracy(3))
        eval_data["top_5_service"].append(s_evaluator.accuracy(5))
        eval_data["avg@5_service"].append(s_evaluator.average(5))
        eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
        eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
        eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
        eval_data["avg@5_metric"].append(f_evaluator.average(5))


# Enhanced evaluation results with fault type grouping (matching experiment.py format)
print("\n" + "="*60)
print(f"📊 Experiment Summary: {args.method} on {args.dataset}")
print("="*60)

# Calculate overall metrics
total_cases = len(rps)
successful_cases = len([rp for rp in rps if not load_json(rp).get("error")])
success_rate = successful_cases / total_cases if total_cases > 0 else 0.0

print(f"📈 Results:")
print(f"  Total cases: {total_cases}")
print(f"  Successful cases: {successful_cases}")
print(f"  Success rate: {success_rate:.1%}")
print(f"  Average speed: {avg_speed:.2f}s per case")
print()

# Calculate overall performance metrics
overall_metrics = {}
metric_names = ['precision@1', 'precision@3', 'precision@5', 'avg@5', 'mrr']

for metric in metric_names:
    if metric == 'precision@1':
        values = [s_evaluator_all.accuracy(1)]
    elif metric == 'precision@3':
        values = [s_evaluator_all.accuracy(3)]
    elif metric == 'precision@5':
        values = [s_evaluator_all.accuracy(5)]
    elif metric == 'avg@5':
        values = [s_evaluator_all.average(5)]
    elif metric == 'mrr':
        # Calculate MRR manually since Evaluator doesn't have mrr() method
        mrr_values = []
        for case_ranks in s_evaluator_all._ranks:
            for i, node in enumerate(case_ranks, 1):
                # This is a simplified MRR calculation
                # In practice, you'd need the ground truth to calculate proper MRR
                mrr_values.append(1.0 / i if i <= 5 else 0.0)
                break  # Only consider first match
        values = [sum(mrr_values) / len(mrr_values) if mrr_values else 0.0]
    
    overall_metrics[metric] = values[0] if values and values[0] is not None else 0.0

# 🔧 調試信息：檢查評估器狀態
print(f"🔍 調試信息:")
print(f"  s_evaluator_all 案例數: {s_evaluator_all.num}")
print(f"  f_evaluator_all 案例數: {f_evaluator_all.num}")
print(f"  實際評估的文件數: {len(rps)}")
print(f"  評估的文件列表: {[basename(rp) for rp in rps]}")
print(f"  data_paths 數量: {len(data_paths)}")
print(f"  data_paths: {[basename(p) for p in data_paths]}")
print(f"  accuracy@1: {s_evaluator_all.accuracy(1)}")
print(f"  accuracy@3: {s_evaluator_all.accuracy(3)}")
print(f"  accuracy@5: {s_evaluator_all.accuracy(5)}")
print(f"  average@5: {s_evaluator_all.average(5)}")
print()

print(f"📊 Overall Performance Metrics:")
for metric, value in overall_metrics.items():
    print(f"  {metric}: {value:.4f}")
print()

# Performance metrics by fault type
print(f"🎯 Performance Metrics by Fault Type:")
print()

fault_evaluators = [
    ("cpu", s_evaluator_cpu, f_evaluator_cpu),
    ("mem", s_evaluator_mem, f_evaluator_mem),
    ("disk", s_evaluator_io, f_evaluator_io),
    ("socket", s_evaluator_socket, f_evaluator_socket),
    ("delay", s_evaluator_lat, f_evaluator_lat),
    ("loss", s_evaluator_loss, f_evaluator_loss),
]

for name, s_evaluator, f_evaluator in fault_evaluators:
    display_name = name.upper()
    if name == "disk":
        display_name = "DISK"
    
    print(f"  📊 {display_name} Faults:")
    
    # Calculate metrics for this fault type
    fault_metrics = {}
    fault_metrics['precision@1'] = s_evaluator.accuracy(1) if s_evaluator.accuracy(1) is not None else 0.0
    fault_metrics['precision@3'] = s_evaluator.accuracy(3) if s_evaluator.accuracy(3) is not None else 0.0
    fault_metrics['precision@5'] = s_evaluator.accuracy(5) if s_evaluator.accuracy(5) is not None else 0.0
    fault_metrics['avg@5'] = s_evaluator.average(5) if s_evaluator.average(5) is not None else 0.0
    # Calculate MRR manually for this fault type
    mrr_values = []
    for case_ranks in s_evaluator._ranks:
        for i, node in enumerate(case_ranks, 1):
            mrr_values.append(1.0 / i if i <= 5 else 0.0)
            break  # Only consider first match
    fault_metrics['mrr'] = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
    
    # Display key metrics
    key_metrics = ['precision@1', 'precision@3', 'precision@5', 'avg@5', 'mrr']
    for metric in key_metrics:
        if metric in fault_metrics:
            value = fault_metrics[metric]
            print(f"    {metric}: {value:.4f}")
    
    print()

print("="*60)
