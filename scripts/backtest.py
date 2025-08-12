#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滚动回测脚本：按“截至 t 训练 -> 预测 t+1 -> 与真值比对”评估推荐管线

指标：
- 红球 Hit@k（等价于 Recall@k）：对每期，统计真值 6 个红球中有多少落在概率 Top-k 中，/6 后对所有期求平均
- 蓝球命中率@k：真值蓝球是否落在蓝球概率 Top-k 内
- 同时输出仅用“时间衰减频率”的基线(Baseline) 与 LSTM 融合(ML) 的对比

用法示例（在仓库根目录）：
    python3 scripts/backtest.py --start 2015-01-01 --seq-len 10 --epochs 3 --k 6 10 12 16 --blue-k 1 2 3 4

说明：
- 默认从 data/lottery_data.json 读取历史数据；若不存在则调用 analyzer 抓取一次后再跑
- 为了速度，默认每个 t 仅训练少量 epoch，可通过 --epochs 调整
- 训练集使用截至 t 的全部历史；目标为第 t+1 期
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np

# 允许相对导入同目录脚本
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
if CUR_DIR not in sys.path:
    sys.path.insert(0, CUR_DIR)

from lottery_analyzer import DoubleColorBallAnalyzer  # noqa: E402

# ---------------------- 工具函数 ----------------------

def sort_records_asc(records):
    # 按日期、期号升序
    return sorted(records, key=lambda r: (r["date"], r["period"]))


def filter_by_start_date(records, start_date_str=None):
    if not start_date_str:
        return records
    try:
        start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except Exception:
        return records
    out = [r for r in records if datetime.strptime(r["date"], "%Y-%m-%d").date() >= start]
    return out


def hit_at_k_red(p_red, true_reds, k):
    """返回该期的 Recall@k：真值 6 个红球有多少出现在 top-k 概率内 / 6."""
    top_idx = np.argsort(p_red)[-k:]  # 索引从0开始，代表号码1..33
    top_set = set(int(i) + 1 for i in top_idx)
    hits = sum(1 for r in true_reds if r in top_set)
    return hits / 6.0


def blue_hit_at_k(p_blue, true_blue, k):
    top_idx = np.argsort(p_blue)[-k:]
    top_set = set(int(i) + 1 for i in top_idx)
    return 1.0 if int(true_blue) in top_set else 0.0


# ---------------------- 回测逻辑 ----------------------

def roll_once(train_records, target_record, seq_len, epochs, lr, hidden_size, use_ml=True):
    """给定训练集与目标期，返回 ML 融合与 Baseline 的两组概率 (p_red, p_blue)."""
    analyzer = DoubleColorBallAnalyzer()
    analyzer.lottery_data = list(train_records)  # 顺序无所谓，内部会排序
    if use_ml:
        try:
            analyzer.train_ml_model(seq_len=seq_len, epochs=epochs, lr=lr, hidden_size=hidden_size)
        except Exception as e:
            print(f"[warn] 训练失败，退化为基线：{e}")
            analyzer.trained = False
    # ML 融合
    p_red_ml, p_blue_ml = analyzer.predict_next_probabilities(blend_alpha=0.6, decay_half_life=60)
    # Baseline（关闭训练后再算一次）
    analyzer.trained = False
    p_red_base, p_blue_base = analyzer.predict_next_probabilities(blend_alpha=0.0, decay_half_life=60)
    return (p_red_ml, p_blue_ml), (p_red_base, p_blue_base)


def backtest(records, seq_len=10, epochs=3, lr=1e-3, hidden_size=64, k_list=(6, 10, 12, 16), blue_k_list=(1, 2, 3, 4), use_ml=True):
    recs = sort_records_asc(records)
    if len(recs) <= seq_len + 1:
        raise ValueError("数据量过少，无法回测")

    # 累积器：每个 k 分别累计
    red_hits_ml = {k: [] for k in k_list}
    red_hits_base = {k: [] for k in k_list}
    blue_hits_ml = {k: [] for k in blue_k_list}
    blue_hits_base = {k: [] for k in blue_k_list}

    # 从 t = seq_len 到 N-2（预测 t+1）
    for t in range(seq_len, len(recs) - 1):
        train = recs[:t]
        target = recs[t]
        (p_red_ml, p_blue_ml), (p_red_base, p_blue_base) = roll_once(
            train, target, seq_len, epochs, lr, hidden_size, use_ml=use_ml
        )
        # 真值
        true_reds = target["red_balls"]
        true_blue = target["blue_ball"]

        for k in k_list:
            red_hits_ml[k].append(hit_at_k_red(p_red_ml, true_reds, min(k, 33)))
            red_hits_base[k].append(hit_at_k_red(p_red_base, true_reds, min(k, 33)))
        for bk in blue_k_list:
            blue_hits_ml[bk].append(blue_hit_at_k(p_blue_ml, true_blue, min(bk, 16)))
            blue_hits_base[bk].append(blue_hit_at_k(p_blue_base, true_blue, min(bk, 16)))

    # 汇总
    def avg(d):
        return {k: (float(np.mean(v)) if len(v) else 0.0) for k, v in d.items()}

    return {
        "N_eval": len(recs) - 1 - seq_len,
        "red_ML": avg(red_hits_ml),
        "red_BASE": avg(red_hits_base),
        "blue_ML": avg(blue_hits_ml),
        "blue_BASE": avg(blue_hits_base),
    }


def pretty_print(summary, k_list, blue_k_list):
    N = summary["N_eval"]
    print("\n================ 回测结果汇总 ================")
    print(f"评估期数: {N}")
    print("-- 红球 Recall@k (每期命中占6的比例的平均) --")
    header = "k\tML\t\tBaseline"
    print(header)
    for k in k_list:
        ml = summary["red_ML"].get(k, 0.0)
        bs = summary["red_BASE"].get(k, 0.0)
        print(f"@{k}\t{ml:.4f}\t{bs:.4f}")
    print("-- 蓝球 命中率@k --")
    for k in blue_k_list:
        ml = summary["blue_ML"].get(k, 0.0)
        bs = summary["blue_BASE"].get(k, 0.0)
        print(f"@{k}\t{ml:.4f}\t{bs:.4f}")
    print("============================================\n")


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="双色球滚动回测")
    ap.add_argument("--data", default="data/lottery_data.json", help="历史数据文件路径；若不存在将自动抓取")
    ap.add_argument("--start", default=None, help="起始日期(含)，格式YYYY-MM-DD，用于裁剪历史")
    ap.add_argument("--seq-len", type=int, default=10, help="LSTM 序列长度")
    ap.add_argument("--epochs", type=int, default=3, help="每步训练的epoch")
    ap.add_argument("--hidden-size", type=int, default=64, help="LSTM隐藏层大小")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    ap.add_argument("--k", type=int, nargs="*", default=[6, 10, 12, 16], help="红球Hit@k列表")
    ap.add_argument("--blue-k", type=int, nargs="*", default=[1, 2, 3, 4], help="蓝球命中@k列表")
    ap.add_argument("--no-ml", action="store_true", help="只跑基线（时间衰减频率）")
    args = ap.parse_args()

    # 加载数据
    if os.path.exists(args.data):
        with open(args.data, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        print("未找到数据文件，尝试抓取一次……")
        analyzer = DoubleColorBallAnalyzer()
        max_pages = analyzer.get_max_pages()
        analyzer.fetch_lottery_data(max_pages=max_pages)
        analyzer.save_data(args.data)
        records = analyzer.lottery_data

    # 官方 API 返回一般为最新在前；为稳妥采用升序后按日期过滤
    records = sort_records_asc(records)
    records = filter_by_start_date(records, args.start)

    summary = backtest(
        records,
        seq_len=args.seq_len,
        epochs=args.epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        k_list=tuple(args.k),
        blue_k_list=tuple(args.blue_k),
        use_ml=(not args.no_ml),
    )

    pretty_print(summary, args.k, args.blue_k)


if __name__ == "__main__":
    main()