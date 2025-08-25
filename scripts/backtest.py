#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
滚动回测脚本：按“截至 t 训练 -> 预测 t+1 -> 与真值比对”评估推荐管线

指标：
- 红球 Hit@k（等价于 Recall@k）：对每期，统计真值 6 个红球中有多少落在概率 Top-k 中，/6 后对所有期求平均
- 蓝球命中率@k：真值蓝球是否落在蓝球概率 Top-k 内
- 同时输出仅用“时间衰减频率”的基线(Baseline) 与 LSTM 融合(ML) 的对比

用法示例（在仓库根目录）：
        python3 scripts/backtest.py \
        --start 2015-01-01 \
        --seq-len 10 --epochs 3 \
        --k 6 10 12 16 \
        --blue-k 1 2 3 4 \
        --half-life 60 \
        --alpha-fixed 0.35 \
        --short-win 30 --long-win 180 --mix-betas 0.2 0.35 0.5

说明：
- 默认从 data/lottery_data.json 读取历史数据；若不存在则调用 analyzer 抓取一次后再跑
- 为了速度，默认每个 t 仅训练少量 epoch，可通过 --epochs 调整
- 训练集使用截至 t 的全部历史；目标为第 t+1 期
"""

# 解析命令行参数
import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# 默认参数（可通过 CLI 覆盖）
DEFAULT_DECAY_HALF_LIFE = 60
DEFAULT_ALPHA_FIXED = 0.40

# 多尺度窗口与融合权重默认
DEFAULT_SHORT_WIN = 30
DEFAULT_LONG_WIN = 180
DEFAULT_MIX_BETAS = [0.20, 0.35, 0.50]

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
    out = [r for r in records if datetime.strptime(
        r["date"], "%Y-%m-%d").date() >= start]
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


# ---------------------- Baseline 计算辅助 ----------------------


def _baseline_probs(
    analyzer, train_records, *, half_life, cond_weekday=None, window=None
):
    """基于给定训练集/窗口计算先验概率。
    如果提供 window，则仅使用最近 window 期记录；否则使用全部 train_records。
    cond_weekday: None 表示全局；否则使用 analyzer 的 compute_marginal_probs 的周几条件版本。
    返回 (p_red, p_blue) float32。
    """
    # 按日期升序
    recs = sort_records_asc(train_records)
    if window is not None and window > 0:
        recs = recs[-int(window):]
    tmp = DoubleColorBallAnalyzer()
    tmp.lottery_data = list(recs)
    pr, pb = tmp.compute_marginal_probs(
        decay_half_life=half_life, cond_weekday=cond_weekday
    )
    return pr, pb


def _mix_probs(p_a, p_b, beta):
    """概率线性混合并归一化: beta * a + (1-beta) * b"""
    a = np.asarray(p_a, dtype=np.float64)
    b = np.asarray(p_b, dtype=np.float64)
    q = beta * a + (1.0 - beta) * b
    s = float(q.sum())
    if s <= 0:
        return (a + b) * 0.5
    return (q / s).astype(np.float32)


# ---------------------- 回测逻辑 ----------------------


def roll_once(
    train_records, target_record, seq_len, epochs, lr, hidden_size, use_ml=True
):
    """给定训练集与目标期，返回多种对比概率。
    返回 dict:
        {
          'ML_auto': (p_red, p_blue),
          'ML_fixed': (p_red, p_blue),
          'BASE_global': (p_red, p_blue),
          'BASE_weekday': (p_red, p_blue),
          'BASE_short': (p_red, p_blue),
          'BASE_long': (p_red, p_blue),
          'BASE_mix_0.20': (p_red, p_blue),
          'BASE_mix_0.35': (p_red, p_blue),
          'BASE_mix_0.50': (p_red, p_blue),
        }
    说明：
    - BASE_global 直接使用全局时间衰减频率（不按周几条件）。
    - BASE_weekday 使用按“下一期开奖日的星期”条件的时间衰减频率（与 analyzer 新逻辑一致）。
    - ML_* 使用 LSTM+先验融合；auto 表示 alpha 自适应，fixed 使用给定 alpha。
    """
    analyzer = DoubleColorBallAnalyzer()
    analyzer.lottery_data = list(train_records)  # 顺序无所谓，内部会排序

    # ============ 训练（可选） ============
    if use_ml:
        try:
            analyzer.train_ml_model(
                seq_len=seq_len, epochs=epochs, lr=lr, hidden_size=hidden_size
            )
        except Exception as e:
            print(f"[warn] 训练失败，退化为基线：{e}")
            analyzer.trained = False

    # 目标开奖星期（用于条件先验）
    target_wd = analyzer._next_draw_weekday()

    # ---------- Baselines ----------
    # Global (不按周几 / 按周几)
    pr_g, pb_g = _baseline_probs(
        analyzer,
        train_records,
        half_life=DEFAULT_DECAY_HALF_LIFE,
        cond_weekday=None,
        window=None,
    )
    pr_wd, pb_wd = _baseline_probs(
        analyzer,
        train_records,
        half_life=DEFAULT_DECAY_HALF_LIFE,
        cond_weekday=target_wd,
        window=None,
    )

    # Short / Long （都使用按周几的条件，便于与模型一致；也可按需切换为 None）
    pr_s, pb_s = _baseline_probs(
        analyzer,
        train_records,
        half_life=DEFAULT_DECAY_HALF_LIFE,
        cond_weekday=target_wd,
        window=DEFAULT_SHORT_WIN,
    )
    pr_l, pb_l = _baseline_probs(
        analyzer,
        train_records,
        half_life=DEFAULT_DECAY_HALF_LIFE,
        cond_weekday=target_wd,
        window=DEFAULT_LONG_WIN,
    )

    # Mix(β) = β*Short + (1-β)*Global_weekday（这里选用 weekday 版的全局以保持条件一致）
    pr_mix = {}
    pb_mix = {}
    for beta in DEFAULT_MIX_BETAS:
        pr_mix[beta] = _mix_probs(pr_s, pr_wd, beta)
        pb_mix[beta] = _mix_probs(pb_s, pb_wd, beta)

    # ---------- ML (auto/fixed) ----------
    ml_auto_red, ml_auto_blue = analyzer.predict_next_probabilities(
        blend_alpha="auto", decay_half_life=DEFAULT_DECAY_HALF_LIFE
    )
    ml_fixed_red, ml_fixed_blue = analyzer.predict_next_probabilities(
        blend_alpha=DEFAULT_ALPHA_FIXED, decay_half_life=DEFAULT_DECAY_HALF_LIFE)

    out = {
        "ML_auto": (ml_auto_red, ml_auto_blue),
        "ML_fixed": (ml_fixed_red, ml_fixed_blue),
        "BASE_global": (pr_g, pb_g),
        "BASE_weekday": (pr_wd, pb_wd),
        "BASE_short": (pr_s, pb_s),
        "BASE_long": (pr_l, pb_l),
    }
    for beta in DEFAULT_MIX_BETAS:
        key = f"BASE_mix_{beta:.2f}"
        out[key] = (pr_mix[beta], pb_mix[beta])
    return out


def backtest(
    records,
    seq_len=10,
    epochs=3,
    lr=1e-3,
    hidden_size=64,
    k_list=(6, 10, 12, 16),
    blue_k_list=(1, 2, 3, 4),
    use_ml=True,
):
    recs = sort_records_asc(records)
    if len(recs) <= seq_len + 1:
        raise ValueError("数据量过少，无法回测")

    variants = [
        "ML_auto",
        "ML_fixed",
        "BASE_global",
        "BASE_weekday",
        "BASE_short",
        "BASE_long",
    ] + [f"BASE_mix_{b:.2f}" for b in DEFAULT_MIX_BETAS]
    # 累积器：每个 k 分别累计
    red_hits = {v: {k: [] for k in k_list} for v in variants}
    blue_hits = {v: {k: [] for k in blue_k_list} for v in variants}

    # 从 t = seq_len 到 N-2（预测 t+1）
    for t in range(seq_len, len(recs) - 1):
        train = recs[:t]
        target = recs[t]
        prob_map = roll_once(
            train, target, seq_len, epochs, lr, hidden_size, use_ml=use_ml
        )

        true_reds = target["red_balls"]
        true_blue = target["blue_ball"]

        for name, (p_red, p_blue) in prob_map.items():
            for k in k_list:
                red_hits[name][k].append(
                    hit_at_k_red(
                        p_red, true_reds, min(
                            k, 33)))
            for bk in blue_k_list:
                blue_hits[name][bk].append(
                    blue_hit_at_k(p_blue, true_blue, min(bk, 16))
                )

    # 汇总
    def avg(d):
        return {k: (float(np.mean(v)) if len(v) else 0.0)
                for k, v in d.items()}

    summary = {"N_eval": len(recs) - 1 - seq_len}
    summary.update({f"red_{name}": avg(series)
                   for name, series in red_hits.items()})
    summary.update({f"blue_{name}": avg(series)
                   for name, series in blue_hits.items()})
    return summary


def pretty_print(summary, k_list, blue_k_list):
    N = summary["N_eval"]
    print("\n================ 回测结果汇总 ================")
    print(f"评估期数: {N}")

    # 红球表
    print("-- 红球 Recall@k (每期命中占6的比例的平均) --")
    base_cols = ["BASE_global", "BASE_weekday", "BASE_short", "BASE_long"] + [
        f"BASE_mix_{b:.2f}" for b in DEFAULT_MIX_BETAS
    ]
    header = "k\tML(auto)\tML(fixed)\t" + "\t".join(base_cols)
    print(header)
    for k in k_list:
        cells = [
            f"@{k}",
            f"{summary.get('red_ML_auto', {}).get(k, 0.0):.4f}",
            f"{summary.get('red_ML_fixed', {}).get(k, 0.0):.4f}",
        ]
        for col in base_cols:
            cells.append(f"{summary.get('red_'+col, {}).get(k, 0.0):.4f}")
        print("\t".join(cells))

    # 蓝球表
    print("-- 蓝球 命中率@k --")
    header_b = "k\tML(auto)\tML(fixed)\t" + "\t".join(base_cols)
    print(header_b)
    for k in blue_k_list:
        cells = [
            f"@{k}",
            f"{summary.get('blue_ML_auto', {}).get(k, 0.0):.4f}",
            f"{summary.get('blue_ML_fixed', {}).get(k, 0.0):.4f}",
        ]
        for col in base_cols:
            cells.append(f"{summary.get('blue_'+col, {}).get(k, 0.0):.4f}")
        print("\t".join(cells))
    print("============================================\n")


# ---------------------- CLI ----------------------


def main():
    global DEFAULT_DECAY_HALF_LIFE, DEFAULT_ALPHA_FIXED
    global DEFAULT_SHORT_WIN, DEFAULT_LONG_WIN, DEFAULT_MIX_BETAS

    # 创建解析器
    ap = argparse.ArgumentParser(description="双色球滚动回测")
    ap.add_argument(
        "--data",
        default="data/lottery_data.json",
        help="历史数据文件路径；若不存在将自动抓取",
    )
    ap.add_argument(
        "--start", default=None, help="起始日期(含)，格式YYYY-MM-DD，用于裁剪历史"
    )
    ap.add_argument("--seq-len", type=int, default=10, help="LSTM 序列长度")
    ap.add_argument("--epochs", type=int, default=3, help="每步训练的epoch")
    ap.add_argument("--hidden-size", type=int, default=64, help="LSTM隐藏层大小")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    ap.add_argument(
        "--k", type=int, nargs="*", default=[6, 10, 12, 16], help="红球Hit@k列表"
    )
    ap.add_argument(
        "--blue-k", type=int, nargs="*", default=[1, 2, 3, 4], help="蓝球命中@k列表"
    )
    ap.add_argument("--no-ml", action="store_true", help="只跑基线（时间衰减频率）")
    ap.add_argument(
        "--half-life",
        type=int,
        default=DEFAULT_DECAY_HALF_LIFE,
        help="时间衰减的半衰期(期数)",
    )
    ap.add_argument(
        "--alpha-fixed",
        type=float,
        default=DEFAULT_ALPHA_FIXED,
        help="ML 固定融合系数 alpha",
    )
    ap.add_argument(
        "--short-win",
        type=int,
        default=DEFAULT_SHORT_WIN,
        help="短窗口期数（用于短期先验）",
    )
    ap.add_argument(
        "--long-win",
        type=int,
        default=DEFAULT_LONG_WIN,
        help="长窗口期数（用于长期先验）",
    )
    ap.add_argument(
        "--mix-betas",
        type=float,
        nargs="*",
        default=DEFAULT_MIX_BETAS,
        help="短期/全局先验融合权重β（可多值）",
    )
    args = ap.parse_args()

    DEFAULT_DECAY_HALF_LIFE = int(args.half_life)
    DEFAULT_ALPHA_FIXED = float(args.alpha_fixed)

    DEFAULT_SHORT_WIN = int(args.short_win)
    DEFAULT_LONG_WIN = int(args.long_win)
    DEFAULT_MIX_BETAS = [float(x) for x in args.mix_betas]

    print(
        f"[cfg] half_life={DEFAULT_DECAY_HALF_LIFE}, alpha_fixed={DEFAULT_ALPHA_FIXED}, short_win={DEFAULT_SHORT_WIN}, long_win={DEFAULT_LONG_WIN}, mix_betas={DEFAULT_MIX_BETAS}"
    )

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
