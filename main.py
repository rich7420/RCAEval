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
        ht, 
        rcd, 
        mmrcd,
        get_gnn_kan_rca  # 延遲導入函數
    )
    # 只在需要時導入 GNN-KAN
    def get_gnn_kan():
        return get_gnn_kan_rca()

elif is_py38():
    from RCAEval.e2e import dummy, e_diagnosis, ht, rcd, mmrcd
    
    def get_gnn_kan():
        raise ImportError("GNN-KAN requires Python 3.10+")
else:
    print("Please use Python 3.8 or 3.10")
    exit(1)

try:
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from RCAEval.e2e.causalrca import causalrca
except ImportError:
    pass


def update_eval_config(
    data_root_path: str, inject_time: int, duration: int = 10, **kwargs
) -> dict:
    return {
        "data_root_path": data_root_path,
        "inject_time": inject_time,
        "duration": duration,
        **kwargs,
    }


def get_data_path(args):
    if args.dataset == "ob":
        return "data/online-boutique"
    elif args.dataset == "mm-ob":
        return "data/online-boutique"
    elif args.dataset == "ss1":
        return "data/sock-shop-1"
    elif args.dataset == "mm-ss1":
        return "data/sock-shop-1"
    elif args.dataset == "ss2":
        return "data/sock-shop-2"
    elif args.dataset == "mm-ss2":
        return "data/sock-shop-2"
    elif args.dataset == "tt":
        return "data/train-ticket"
    elif args.dataset == "mm-tt":
        return "data/train-ticket"
    elif args.dataset == "re1":
        return "data/re-1"
    elif args.dataset == "re2-ob":
        return "data/re2/ob"
    elif args.dataset == "re2-tt":
        return "data/re2/tt"
    elif args.dataset == "re3":
        return "data/re-3"
    elif args.dataset == "synthetic":
        return "data/synthetic"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def run_single_experiment(args, fail_id, sub_args):
    data_path = os.path.join(get_data_path(args), str(fail_id))

    # 根據方法決定是否導入 GNN-KAN
    if args.method == "gnn_kan":
        try:
            gnn_kan = get_gnn_kan()
            # 🎯 應用優化參數（如果來自比較場景）
            optimized_config = {
                'graph_head': 'pagerank',
                'config_type': 'simplified',
                'feature_method': 'kpca',
                'kpca_kernel': 'rbf',
                'learning_rate': 9e-5,
                'num_epochs': 200,
                'sparsity_lambda': 1e-4,
                'use_cuda': True,
                'cpu_fallback': True,
                'use_optimized_input': True,
                'similarity_threshold': 0.15,
                'max_edges_per_node': 12,
                'target_feature_dim': 64,
                'force_node_expansion': True
            }
            # 將優化參數注入到 sub_args
            sub_args.update(optimized_config)
            print(f"🎯 Applied optimized GNN-KAN parameters for comparison")
        except ImportError as e:
            print(f"Failed to import GNN-KAN: {e}")
            return None
    else:
        gnn_kan = None

    try:
        if "mm-" in args.dataset:
            # Multi-source dataset handling
            from RCAEval.utility import read_multimodal_data
            
            data, inject_time, meta = read_multimodal_data(data_path, **sub_args)
        else:
            # Single-source dataset handling  
            from RCAEval.utility import read_data

            data, inject_time, meta = read_data(data_path, **sub_args)

        # Execute the RCA method
        if args.method == "gnn_kan" and gnn_kan is not None:
            result = gnn_kan(data, inject_time, dataset=args.dataset, **sub_args)
        else:
            # Use globals() for other methods
            result = globals()[args.method](data, inject_time, dataset=args.dataset, **sub_args)

        # Prepare evaluation configuration
        eval_config = update_eval_config(
            data_path, inject_time, meta.get("duration", 10), **sub_args
        )

        # Evaluate results
        evaluator = Evaluator(config=eval_config)
        evaluation_result = evaluator.eval(result, meta)

        return {
            "result": result,
            "evaluation": evaluation_result,
            "meta": meta,
            "config": eval_config,
        }

    except Exception as e:
        print(f"Error in experiment {fail_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 動態構建可用方法列表
    available_methods = []
    for name in globals():
        if callable(globals()[name]) and not name.startswith('_'):
            available_methods.append(name)
    
    # 添加 gnn_kan 到可用方法列表
    available_methods.append('gnn_kan')
    
    # 移除不是 RCA 方法的函數
    method_blacklist = [
        'parse_args', 'get_data_path', 'run_single_experiment', 
        'update_eval_config', 'get_gnn_kan', 'get_gnn_kan_rca',
        'Pool', 'Evaluator', 'Node', 'tqdm'
    ]
    available_methods = [m for m in available_methods if m not in method_blacklist]
    
    parser.add_argument("--method", type=str, required=True, 
                        help=f"Available methods: {sorted(available_methods)}")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--dk_select_useful", action="store_true")
    parser.add_argument("--dk_select_sli", type=str, default=None)
    parser.add_argument("--dk_select_sli_prefix", type=str, default=None)
    # 添加缺失的參數以保持兼容性
    parser.add_argument("--length", type=int, default=None, help="Time series length")
    parser.add_argument("--tdelta", type=int, default=0, help="Time delta for anomaly detection delay")
    parser.add_argument("--iter_num", type=int, default=10, help="Number of iterations")
    parser.add_argument("--useful", action="store_true", help="Select useful columns")
    
    args = parser.parse_args()
    
    # 檢查方法是否可用
    if args.method not in available_methods:
        print(f"Available methods: {sorted(available_methods)}")
        raise ValueError(f"{args.method=} not defined. Please check imported methods.")
    
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
    if args.method == 'gnn_kan':
        func = get_gnn_kan()
    else:
        func = globals()[args.method]

    try:
        st = datetime.now()
        
        # 🚀 GNN-KAN 優化參數應用
        gnn_kan_kwargs = {}
        if args.method == 'gnn_kan':
            print("🚀 使用 GNN-KAN 優化參數...")
            # 應用優化的參數
            gnn_kan_kwargs = {
                'graph_head': 'pagerank',
                'config_type': 'simplified',
                'feature_method': 'kpca',
                'kpca_kernel': 'rbf',
                'learning_rate': 9e-5,
                'num_epochs': 200,
                'sparsity_lambda': 1e-4,
                'use_cuda': True,
                'cpu_fallback': True,
                'use_optimized_input': True,
                'similarity_threshold': 0.15,
                'max_edges_per_node': 12,
                'target_feature_dim': 64,
                'force_node_expansion': True
            }
            print(f"✅ 已應用優化 GNN-KAN 參數")
            
        result = func(
            data,
            inject_time,
            dataset=args.dataset.split("-")[0] if "-" in args.dataset else args.dataset,
            num_loop=args.iter_num,
            sli=sli,
            dk_select_useful=args.dk_select_useful,
            **gnn_kan_kwargs
        )
        root_causes = result.get("ranks")
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
                # Handle ranks that may not follow the expected format
                s_ranks = []
                for x in ranks:
                    parts = x.split("_")
                    service_name = parts[0].replace("-db", "")
                    s_ranks.append(Node(service_name, "unknown"))
                
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

                # Handle ranks for full metric-level evaluation
                f_ranks = []
                for x in ranks:
                    parts = x.split("_")
                    if len(parts) >= 2:
                        service_name = parts[0]
                        metric_name = parts[1]
                    else:
                        # If no underscore or only one part, treat as service with unknown metric
                        service_name = parts[0]
                        metric_name = "unknown"
                    f_ranks.append(Node(service_name, metric_name))

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


print("--- Evaluation results ---")
for name, s_evaluator, f_evaluator in [
    ("cpu", s_evaluator_cpu, f_evaluator_cpu),
    ("mem", s_evaluator_mem, f_evaluator_mem),
    ("io", s_evaluator_io, f_evaluator_io),
    ("socket", s_evaluator_socket, f_evaluator_socket),
    ("delay", s_evaluator_lat, f_evaluator_lat),
    ("loss", s_evaluator_loss, f_evaluator_loss),
]:
    eval_data["service-fault"].append(f"overall_{name}")
    eval_data["top_1_service"].append(s_evaluator.accuracy(1))
    eval_data["top_3_service"].append(s_evaluator.accuracy(3))
    eval_data["top_5_service"].append(s_evaluator.accuracy(5))
    eval_data["avg@5_service"].append(s_evaluator.average(5))
    eval_data["top_1_metric"].append(f_evaluator.accuracy(1))
    eval_data["top_3_metric"].append(f_evaluator.accuracy(3))
    eval_data["top_5_metric"].append(f_evaluator.accuracy(5))
    eval_data["avg@5_metric"].append(f_evaluator.average(5))

    if name == "io":
        name = "disk"

    if s_evaluator.average(5) is not None:
        print( f"Avg@5-{name.upper()}:".ljust(12), round(s_evaluator.average(5), 2))


print("---")
print("Avg speed:", avg_speed)

