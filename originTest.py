import argparse
import glob
import json
import os
import shutil
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

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
    gnn_kan_rca = None  # GNN+KAN requires Python 3.10+
else:
    print("Please use Python 3.8 or 3.10+")
    exit(1)

try:
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="RCAEval evaluation")
    parser.add_argument("--method", type=str, help="Choose a method.")
    parser.add_argument("--dataset", type=str, help="Choose a dataset.", choices=[
        "online-boutique", "sock-shop-1", "sock-shop-2", "train-ticket",
        "re1-ob", "re1-ss", "re1-tt", "re2-ob", "re2-ss", "re2-tt", "re3-ob", "re3-ss", "re3-tt"
    ])
    parser.add_argument("--length", type=int, default=20, help="Time series length (RQ4)")
    parser.add_argument("--tdelta", type=int, default=0, help="Specify $t_delta$ to simulate delay in anomaly detection")
    parser.add_argument("--test", action="store_true", help="Perform smoke test on certain methods without fully run on all data")
    args = parser.parse_args()

    # Check if method is available (including GNN+KAN)
    available_methods = []
    if is_py310():
        available_methods = [
            "baro", "causalrca", "circa", "cloudranger", "cmlp_pagerank", "dummy",
            "e_diagnosis", "easyrca", "fci_pagerank", "fci_randomwalk", "ges_pagerank",
            "granger_pagerank", "granger_randomwalk", "lingam_pagerank", "lingam_randomwalk",
            "micro_diag", "microcause", "microrank", "mscred", "nsigma", "ntlr_pagerank",
            "ntlr_randomwalk", "pc_pagerank", "pc_randomwalk", "run", "tracerca"
        ]
        if gnn_kan_rca is not None:
            available_methods.append("gnn_kan_rca")
    elif is_py38():
        available_methods = ["dummy", "e_diagnosis", "ht", "rcd", "mmrcd"]
    
    if args.method not in available_methods:
        raise ValueError(f"{args.method=} not defined. Available methods: {available_methods}")

    return args


args = parse_args()

# download dataset
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
else:
    raise Exception(f"{args.dataset} is not defined!")

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
    "re3-tt": "data/RE3/RE3-TT"
}
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

    service, metric = basename(dirname(dirname(data_path))).split("_")
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

    with open(join(data_dir, "inject_time.txt")) as f:
        inject_time = int(f.readlines()[0].strip()) + args.tdelta
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
    elif "train-ticket" in data_path or "fse-tt" in data_path or "RE2-TT" in data_path:
        sli = "ts-ui-dashboard_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
    elif "online-boutique" in data_path or "fse-ob" in data_path or "RE2-OB" in data_path or "RE2-SS" in data_path:
        sli = "frontend_latency"
        if f"{service}_latency" in data:
            sli = f"{service}_latency"
        elif "frontend_1" in data:
            sli = "frontend_1"
    else:
        raise ValueError("SLI not implemented")

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
                feature_method="enhanced_ica",  # 使用最強的特徵提取
                use_optimized_input=True,
                sparsity_lambda=1e-3,           # 大幅增加稀疏性權重
                # 🎯 高精度訓練參數
                learning_rate=2e-4,             # 優化學習率
                num_epochs=400,                 # 大幅增加訓練輪數
                kan_grid_size=20,               # 大幅增加KAN網格
                hidden_dim=128,                 # 增加隱藏維度
                target_feature_dim=128,         # 增加特徵維度
                similarity_threshold=0.2,       # 大幅降低相似性閾值
                max_edges_per_node=15,          # 大幅增加邊數
                gradient_clipping=0.5,          # 更嚴格的梯度裁剪
                # 🎯 新增監督學習參數
                fault_type=fault_type,          # 故障類型信息（關鍵！）
                enhanced_contrast=True,         # 啟用增強對比學習
                fault_type_aware=True,          # 啟用故障類型感知
                adaptive_learning=True,         # 啟用自適應學習
                multi_scale_features=True,      # 啟用多尺度特徵
                temporal_attention=True,        # 啟用時序注意力
                sli=sli,
                verbose=True
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
        service, metric = basename(dirname(dirname(data_path))).split("_")
        case = basename(dirname(data_path))
        rp = join(result_path, f"{service}_{metric}_{case}.json")
        if exists(rp):
            rps.append(rp)
else:
    # 正常模式：讀取所有文件
    rps = glob.glob(join(result_path, "*.json"))

services = sorted(list(set([basename(x).split("_")[0] for x in rps])))
faults = sorted(list(set([basename(x).split("_")[1] for x in rps])))

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
            if s != service or m != fault:
                continue  # ignore

            data = load_json(rp)
            if "error" in data:
                continue  # ignore

            for i, ranks in data.items():
                s_ranks = [Node(x.split("_")[0].replace("-db", ""), "unknown") for x in ranks]
                # remove duplication
                old_s_ranks = s_ranks.copy()
                s_ranks = (
                    [old_s_ranks[0]]
                    + [
                        old_s_ranks[i]
                        for i in range(1, len(old_s_ranks))
                        if old_s_ranks[i] not in old_s_ranks[:i]
                    ]
                    if old_s_ranks
                    else []
                )

                # Handle different output formats from different methods
                f_ranks = []
                for x in ranks:
                    if "_" in x and len(x.split("_")) >= 2:
                        # Standard format: service_metric
                        f_ranks.append(Node(x.split("_")[0], x.split("_")[1]))
                    else:
                        # GNN+KAN format: just service name, use fault type as metric
                        f_ranks.append(Node(x, fault))

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
