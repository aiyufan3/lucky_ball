#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快乐8（Keno）数据抓取、统计分析、推荐与报告生成脚本

用法（本地运行示例）:
python scripts/super_eight.py --fetch --limit 30

python scripts/super_eight.py \
  --fetch --limit 30 \
  --recommend 5 \
  --play_pick 7 \
  --kelly_mode ignore \
  --payouts data/payouts_kl8.json \
  --plan --budget 22 --price_per_bet 2 \
  --plots --plots_dual --split 40 \
  --beta 8 --max_share 0.6 --min_stake 1 \
  --report --seed 42

说明:
- 数据来源: 中国福利彩票官网统一接口 findDrawNotice (name=kl8)
- 仅用于学习与研究，切勿用于任何形式的博彩或引导投注。

依赖: requests, numpy (可选，若未安装将退化为纯 Python 实现)
如果需要生成图表，请自行在项目中添加可视化模块（例如 matplotlib）并扩展本脚本。
"""
from __future__ import annotations
import argparse
import collections
import datetime as dt
import json
import math
from math import comb
import os
import random
import statistics
import time
from typing import Dict, List, Tuple
import itertools
from statistics import mean, median
from matplotlib.ticker import MaxNLocator

# 尝试导入 numpy，若失败则允许无 numpy 环境运行
try:
    import numpy as np
except Exception:
    np = None  # 允许无 numpy 环境运行

import requests

import matplotlib
matplotlib.use("Agg")  # for headless environments
import matplotlib.pyplot as plt
# 中文字体与负号（跨平台回退）
# 依次尝试常见中文字体，最后退回 DejaVu Sans 以避免报错
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = [
    'PingFang SC',            # macOS
    'Hiragino Sans GB',       # macOS (部分)
    'Heiti SC',               # macOS 旧版
    'Microsoft YaHei',        # Windows
    'SimHei',                 # Windows/Linux
    'Noto Sans CJK SC',       # 多平台
    'WenQuanYi Zen Hei',      # Linux
    'Arial Unicode MS',       # 备选
    'DejaVu Sans'             # 最后兜底（英文+部分符号）
]
plt.rcParams['axes.unicode_minus'] = False

API_URL = (
    "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, "kl8_history.json")
REPORT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports/kl8_analysis_report.md")

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pics")
os.makedirs(PLOTS_DIR, exist_ok=True)
DUAL_FREQ_FILE = os.path.join(PLOTS_DIR, "kl8_dual_frequency_style.png")
DEFAULT_PAYOUTS_FILE = os.path.join(DATA_DIR, "payouts_kl8.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )
}

# ----------------------------- 数据抓取 ----------------------------- #

def fetch_kl8_history(limit: int | None = None, sleep_sec: float = 0.1) -> List[dict]:
    """从官方接口增量抓取快乐8历史数据。

    Args:
        limit: 期数上限（None 表示尽可能多地分页抓取）。当提供 limit 时将严格返回最近 limit 期。
        sleep_sec: 每页之间的最小停顿，避免频繁请求。

    Returns:
        包含每期开奖条目的列表（按时间倒序，即最近在前）。
    """
    params = {
        "name": "kl8",
        "issueCount": "",
        "issueStart": "",
        "issueEnd": "",
        "dayStart": "",
        "dayEnd": "",
        "pageNo": "1",
        "pageSize": "100",
        "week": "",
        "systemType": "PC",
    }

    # 如果给了 limit，则只请求第一页并让后端限制返回条数；再在客户端兜底截断
    if limit is not None and limit > 0:
        params["issueCount"] = str(limit)
        params["pageSize"] = str(min(100, limit))
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("state") != 0:
            raise RuntimeError(f"API 返回异常: {payload}")
        results = list(payload.get("result", []))[:limit]
        return results

    # 未指定 limit：分页直到返回为空或不足一页
    results: List[dict] = []
    page = 1
    while True:
        params["pageNo"] = str(page)
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)
        r.raise_for_status()
        payload = r.json()
        if payload.get("state") != 0:
            break
        page_res = list(payload.get("result", []))
        if not page_res:
            break
        results.extend(page_res)
        # 若本页已不足 pageSize，说明到底了
        try:
            ps = int(params.get("pageSize", "100"))
        except Exception:
            ps = 100
        if len(page_res) < ps:
            break
        page += 1
        time.sleep(sleep_sec)
    return results


def normalize_entry(e: dict) -> dict:
    """将接口返回的一条记录标准化。

    期望字段:
        code: 期号 (str)
        date: 开奖日期 (str)
        week: 周几 (str)
        nums: 20 个号码的升序整数列表
    """
    code = str(e.get("code"))
    date = str(e.get("date", ""))
    week = str(e.get("week", ""))

    # 快乐8接口常见字段为 red，逗号分隔 20 个号码
    raw = e.get("red") or e.get("result") or e.get("openCode")
    if not raw:
        raise ValueError(f"未识别的号码字段: {e}")

    if isinstance(raw, list):
        nums = [int(x) for x in raw]
    else:
        nums = [int(x) for x in str(raw).replace(" ", "").split(",") if x]

    nums.sort()
    if len(nums) != 20:
        # 个别省站镜像可能短缺，容错但提示
        # 仍然返回当前解析到的号码，便于后续分析做筛除
        pass

    return {"code": code, "date": date, "week": week, "nums": nums}


def load_existing() -> List[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(entries: List[dict]) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


# ----------------------------- 统计分析 ----------------------------- #

def build_occ_matrix(entries: List[dict], k: int = 80) -> List[List[int]]:
    """构造旧->新（时间向前）的出现矩阵，形状: T x k，元素为0/1。"""
    T = len(entries)
    occ = [[0] * k for _ in range(T)]
    for t, e in enumerate(reversed(entries)):
        for n in e["nums"]:
            if 1 <= n <= k:
                occ[t][n - 1] = 1
    return occ

def ema_matrix(occ: List[List[int]], alpha: float = 0.2) -> List[List[float]]:
    """对 occ 的每个列应用 EMA，返回同形状矩阵，每一行是该时刻的 EMA 值。"""
    if not occ:
        return []
    T, K = len(occ), len(occ[0])
    ema = [[0.0] * K for _ in range(T)]
    # 初始化为第一行
    ema[0] = [float(v) for v in occ[0]]
    for t in range(1, T):
        for c in range(K):
            ema[t][c] = alpha * occ[t][c] + (1 - alpha) * ema[t - 1][c]
    return ema

def frequency_stats(entries: List[dict], k: int = 80) -> Dict[int, int]:
    cnt = collections.Counter()
    for e in entries:
        for n in e["nums"]:
            cnt[n] += 1
    # 补全 1..k
    return {i: cnt.get(i, 0) for i in range(1, k + 1)}


def rolling_ema(freq_series: List[int], alpha: float = 0.2) -> List[float]:
    """对单个号码出现与否的布尔序列进行 EMA，返回同长度的平滑序列。"""
    ema = []
    s = 0.0
    for i, v in enumerate(freq_series):
        s = alpha * v + (1 - alpha) * (s if i > 0 else v)
        ema.append(s)
    return ema


def trend_weights(entries: List[dict], k: int = 80, alpha: float = 0.2) -> List[float]:
    """基于最近期趋势（EMA）的号码权重，越近越大。"""
    # 构造时间顺序（旧->新）的出现布尔矩阵: T x k
    occ = build_occ_matrix(entries, k=k)
    T = len(occ)
    # 对每列做 EMA, 取最后一个值作为近期热度
    weights = []
    for col in range(k):
        series = [occ[t][col] for t in range(T)]
        ema = rolling_ema(series, alpha=alpha)
        weights.append(ema[-1] if ema else 0.0)
    # 防止全 0
    s = sum(weights)
    if s == 0:
        return [1.0] * k
    return [w / s for w in weights]


def sample_without_replacement(weights: List[float], m: int, rng: random.Random) -> List[int]:
    """按权重不放回采样 m 个不同号码（1..len(weights)）。"""
    idxs = list(range(1, len(weights) + 1))
    w = list(weights)
    chosen = []
    for _ in range(m):
        # 简单 roulette 轮盘法
        total = sum(w)
        if total <= 0:
            # 退化为均匀
            pick = rng.choice(idxs)
        else:
            r = rng.random() * total
            acc = 0.0
            pick = None
            for i, (idn, wi) in enumerate(zip(idxs, w)):
                acc += wi
                if acc >= r:
                    pick = idn
                    # 移除该元素（不放回）
                    idxs.pop(i)
                    w.pop(i)
                    break
            if pick is None:
                pick = idxs.pop()
                w.pop()
        chosen.append(pick)
    chosen.sort()
    return chosen


def recommend_sets(entries: List[dict], sets: int = 5, rng_seed: int | None = None) -> List[List[int]]:
    """基于频率+EMA 趋势的混合权重进行推荐。

    约束:
      - 号码区间覆盖: 1-20, 21-40, 41-60, 61-80 四象限尽量均衡
      - 奇偶平衡: 期望约 10:10, 允许在 ±2 范围
    """
    k = 80
    rng = random.Random(rng_seed)

    # 频率与趋势融合
    freq = frequency_stats(entries, k=k)
    freq_weights = [freq[i] + 1e-6 for i in range(1, k + 1)]  # +eps 防 0
    # 归一化
    s = float(sum(freq_weights))
    freq_weights = [x / s for x in freq_weights]

    trend_w = trend_weights(entries, k=k, alpha=0.25)

    # 融合: w = 0.6 * trend + 0.4 * freq
    weights = [0.6 * tw + 0.4 * fw for tw, fw in zip(trend_w, freq_weights)]

    recs: List[List[int]] = []
    attempts = 0
    while len(recs) < sets and attempts < sets * 20:
        attempts += 1
        pick = sample_without_replacement(weights, 20, rng)
        # 约束过滤
        buckets = [sum(1 for x in pick if a <= x <= b) for a, b in [(1, 20), (21, 40), (41, 60), (61, 80)]]
        odd = sum(1 for x in pick if x % 2 == 1)
        even = 20 - odd
        if max(buckets) - min(buckets) > 5:
            continue
        if abs(odd - even) > 4:
            continue
        recs.append(pick)
    return recs


# ----------------------------- 收益方案（资金分配） ----------------------------- #
def score_set_by_weights(pick: List[int], weights: List[float]) -> float:
    """用当前号码权重对一组pick打分：简单求和（可替换为logit/乘积）。"""
    return float(sum(weights[i - 1] for i in pick if 1 <= i <= len(weights)))

def pick_weight_score(pick: List[int], weights: List[float]) -> float:
    """别名：号码组的权重得分（与 score_set_by_weights 相同）。"""
    return score_set_by_weights(pick, weights)
def allocate_budget_by_score(recs: List[List[int]], weights: List[float], total_budget: float, price_per_bet: float = 2.0) -> List[dict]:
    """
    按权重分数进行资金分配（比例法，至少1注），返回每组的计划：
    [{'index':1, 'pick':[...], 'score':..., 'stakes': 注数, 'amount': 金额}]
    """
    if total_budget <= 0 or price_per_bet <= 0 or not recs:
        return []
    scores = [max(score_set_by_weights(r, weights), 1e-9) for r in recs]
    s = sum(scores)
    # 先按比例计算期望注数，再向下取整，剩余按分数排序逐一补1注
    expected = [(sc / s) * (total_budget / price_per_bet) for sc in scores]
    stakes = [int(x) for x in expected]
    # 至少保证每组1注
    stakes = [max(1, n) for n in stakes]
    used = sum(stakes)
    target = int(total_budget // price_per_bet)
    # 调整到不超过预算
    while used > target and any(stakes):
        # 从最低分往下减
        i = stakes.index(max(stakes)) if used - target > 5 else stakes.index(max(stakes))
        stakes[i] -= 1
        if stakes[i] < 0:
            stakes[i] = 0
        used = sum(stakes)
    # 若预算还有余额，按分数高的补齐
    order = sorted(range(len(recs)), key=lambda i: scores[i], reverse=True)
    i_ptr = 0
    while used < target and order:
        stakes[order[i_ptr % len(order)]] += 1
        used += 1
        i_ptr += 1

    plan = []
    for idx, (r, sc, st) in enumerate(zip(recs, scores, stakes), 1):
        plan.append({
            "index": idx,
            "pick": r,
            "score": round(sc, 6),
            "stakes": int(st),
            "amount": round(st * price_per_bet, 2)
        })
    return plan

def write_profit_plan(entries: List[dict], recs: List[List[int]], plan: List[dict], total_budget: float, price_per_bet: float, out_path: str | None = None) -> str:
    """
    生成 'kl8_profit_plan.md'：仅资金分配建议（学习用途）。
    说明：快乐8玩法及赔率多样，若要计算期望收益，请在此项目中补充具体玩法的赔率表，并基于超几何分布计算命中概率与期望值。
    """
    out_path = out_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports/kl8_profit_plan.md")
    lines = []
    lines.append("# 快乐8 最近30期 - 资金分配示例方案（学习用途）\n")
    # 文案：若为EV方案，说明基于“EV+权重分散”；否则为“权重示例”
    if plan and any('ev_per_bet' in it for it in plan):
        lines.append("> 免责声明：彩票随机，历史不代表未来；本方案基于 *玩法EV + 权重分散* 的示例方法，仅用于技术学习，不构成投注建议。\n")
    else:
        lines.append("> 免责声明：彩票随机，历史不代表未来；本方案仅展示一种基于权重打分的资金分配方法，不构成投注建议。\n")
    # 统计本期计划的总注数（若无计划则按预算/单价给出理论注数）
    total_stakes = sum(int(it.get('stakes', 0)) for it in plan) if plan else 0
    planned_stakes = total_stakes if total_stakes > 0 else int(total_budget // price_per_bet)
    lines.append(f"- 总预算：¥{total_budget:.2f} ； 单注价格：¥{price_per_bet:.2f} ； 计划总注数：{planned_stakes}\n")
    if planned_stakes == 0:
        lines.append("- 注：当前 Kelly 严格模式在本玩法/赔率下给出 0 注（负EV或过低），可使用 --kelly_mode floor/ignore 查看非零分配示例。\n")
    # 最近一期
    if entries:
        latest = entries[0]
        lines.append(f"- 最近开奖：期号 {latest.get('code')} | 日期 {latest.get('date')} | 号码：{', '.join(f'{x:02d}' for x in latest.get('nums', []))}\n")

    # 若计划含EV字段，增加说明
    if plan and any('ev_per_bet' in it for it in plan):
        ev0 = next((it['ev_per_bet'] for it in plan if 'ev_per_bet' in it), None)
        if ev0 is not None:
            lines.append(f"- 本玩法单注期望收益(理论，基于赔率与超几何)：{ev0} 元/注\n")
    # 仅当有计划且存在 ev_per_bet 字段且总注数>0 时，输出EV汇总
    if plan and any('ev_per_bet' in it for it in plan):
        ev0 = next((it['ev_per_bet'] for it in plan if 'ev_per_bet' in it), None)
        if ev0 is not None and total_stakes > 0:
            total_ev = round(total_stakes * float(ev0), 2)
            lines.append(f"- 本期计划总注数：{total_stakes} 注；**理论期望盈亏**：{total_ev} 元\n")
    lines.append("\n## 推荐组合与资金分配\n")
    total_amount = 0.0
    for item in plan:
        label = []
        if 'ev_per_bet' in item:
            label.append(f"单注EV {item['ev_per_bet']}")
        if 'score' in item:
            label.append(f"分数 {item['score']}")
        if 'prob' in item:
            label.append(f"概率 {item['prob']}")
        if 'share' in item:
            label.append(f"分配占比 {round(item['share']*100, 2)}%")
        label_str = " | ".join(label) if label else "分数/EV —"
        lines.append(
            f"- 方案{item['index']:02d} | {label_str} | 注数 {item.get('stakes', 0):>3} | 金额 ¥{item.get('amount', 0):.2f} | 号码：{', '.join(f'{x:02d}' for x in item.get('pick', []))}"
        )
        total_amount += float(item.get('amount', 0))
    lines.append(f"\n**合计**：¥{total_amount:.2f}\n")

    lines.append("\n---\n### 如何扩展为“期望收益方案”\n")
    lines.append("1) 指定玩法（例如任选10、任选7等）并给出对应**官方赔率表**；")
    lines.append("2) 用超几何分布计算各命中档位的概率；")
    lines.append("3) 计算每组组合的**期望收益=∑(档位概率×对应奖金) - 注金**；")
    lines.append("4) 采用凯利/风险约束将预算分配给**期望收益为正**或Sharpe较高的组合；")
    lines.append("5) 在报告中展示EV、方差与回测对比随机基线的分布图。")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


# ----------------------------- 回测与蒙特卡洛 ----------------------------- #

def overlap_count(draw_nums: List[int], pick: List[int]) -> int:
    """计算命中数量（与官方开奖的交集大小）。"""
    s = set(draw_nums)
    return sum(1 for x in pick if x in s)

def random_pick(k: int = 80, m: int = 20, rng: random.Random | None = None) -> List[int]:
    r = rng or random.Random()
    nums = r.sample(range(1, k + 1), m)
    nums.sort()
    return nums

def monte_carlo_overlap(actual: List[int], trials: int = 10000, k: int = 80, m: int = 20, rng_seed: int | None = 123) -> List[int]:
    """随机基线（超几何近似的经验分布）：与真实开奖的重叠样本分布。"""
    r = random.Random(rng_seed)
    samples = []
    for _ in range(trials):
        pick = random_pick(k=k, m=m, rng=r)
        samples.append(overlap_count(actual, pick))
    return samples

def backtest_overlap(entries: List[dict], window: int = 200, sets: int = 5, alpha: float = 0.25, rng_seed: int | None = 123, random_trials: int = 2000) -> dict:
    """
    滚动回测：用过去 window 期训练权重→推荐 sets 组→与下一期真实开奖比较命中数；
    同时给出相同评估规模下的随机基线分布。
    返回:
        {
          'model_overlaps': [...],   # 每期取“最佳一组”的命中数
          'random_overlaps': [...],  # 随机基线同样数量的样本
          'summary': {均值/中位数/p90等}
        }
    """
    if len(entries) <= window + 1:
        return {'model_overlaps': [], 'random_overlaps': [], 'summary': {}}

    rng = random.Random(rng_seed)
    model_ov = []
    random_ov = []

    # 逐期评估
    for t in range(window, len(entries) - 1):
        history = entries[t - window: t]  # 不包含 t
        next_draw = entries[t]            # t 这一期作为验证
        weights = trend_weights(history, k=80, alpha=alpha)
        # 生成 sets 组
        picks = [sample_without_replacement(weights, 20, rng) for _ in range(sets)]
        # 取“最佳一组”与真实开奖的重叠
        hits = [overlap_count(next_draw["nums"], p) for p in picks]
        model_ov.append(max(hits))

        # 随机基线：与同一真实开奖对比
        random_ov.extend(monte_carlo_overlap(next_draw["nums"], trials=max(10, random_trials // (len(entries) // max(1, (len(entries) - window)))), k=80, m=20, rng_seed=rng.randint(1, 10**9)))

    # 归一化随机样本数量到与模型次数一致（简单截断/扩展）
    if len(random_ov) < len(model_ov):
        need = len(model_ov) - len(random_ov)
        random_ov += random_ov[:need]
    elif len(random_ov) > len(model_ov):
        random_ov = random_ov[:len(model_ov)]

    def pct(xs, p):
        if not xs:
            return None
        xs2 = sorted(xs)
        idx = int((p / 100.0) * (len(xs2) - 1))
        return xs2[idx]

    summary = {
        "model_mean": round(mean(model_ov), 4) if model_ov else None,
        "model_median": round(median(model_ov), 4) if model_ov else None,
        "model_p90": pct(model_ov, 90),
        "random_mean": round(mean(random_ov), 4) if random_ov else None,
        "random_median": round(median(random_ov), 4) if random_ov else None,
        "random_p90": pct(random_ov, 90),
        "samples": len(model_ov),
        "window": window,
        "sets": sets,
    }
    return {"model_overlaps": model_ov, "random_overlaps": random_ov, "summary": summary}

# ----------------------------- 指标与报告 ----------------------------- #

def basic_metrics(entries: List[dict]) -> Dict[str, float]:
    """计算一些基础指标用于报告展示。"""
    if not entries:
        return {}
    draws = len(entries)
    sums = [sum(e["nums"]) for e in entries if len(e["nums"]) == 20]
    sum_avg = statistics.mean(sums) if sums else float("nan")
    # 期望和值 ~ 20 * (1+80)/2 = 810
    expected_sum = 20 * (1 + 80) / 2
    return {
        "draws": draws,
        "sum_avg": round(sum_avg, 2) if sums else None,
        "sum_expected": expected_sum,
    }


def top_hot_and_cold(entries: List[dict], k: int = 80, topn: int = 10) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    freq = frequency_stats(entries, k=k)
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    hot = items[:topn]
    cold = list(reversed(items))[:topn]
    return hot, cold


def write_report(entries: List[dict], recs: List[List[int]], backtest_summary: Dict[str, float] | None = None) -> None:
    meta = basic_metrics(entries)
    hot, cold = top_hot_and_cold(entries, k=80, topn=15)
    latest = entries[0] if entries else {}

    def fmt_pairs(pairs: List[Tuple[int, int]]):
        return ", ".join([f"{n:02d}({c})" for n, c in pairs])

    md = []
    md.append("# 中国福利彩票 快乐8 分析报告\n")
    md.append("> 免责声明：历史数据无法预测未来。本报告仅供技术学习参考，不构成投注建议。\n\n")
    md.append(f"**数据期数**: {meta.get('draws', 0)} 期\n")
    if latest:
        md.append(f"**最近开奖**: 期号 {latest.get('code')} | 日期 {latest.get('date')} | 号码: {', '.join(f'{x:02d}' for x in latest.get('nums', []))}\n")
    if meta:
        md.append(f"**和值均值(样本)**: {meta.get('sum_avg')} ；**理论期望**: {int(meta.get('sum_expected', 810))}\n")
    md.append("\n## 热号 Top15\n")
    md.append(fmt_pairs(hot) + "\n")
    md.append("\n## 冷号 Top15\n")
    md.append(fmt_pairs(cold) + "\n")

    if recs:
        md.append("\n## 推荐组合（示例，非投注建议）\n")
        for i, r in enumerate(recs, 1):
            md.append(f"- 方案{i}: " + ", ".join(f"{x:02d}" for x in r))

    if backtest_summary:
        md.append("\n## 回测摘要（模型 vs 随机）\n")
        kvs = [
            ("样本数", backtest_summary.get("samples")),
            ("窗口", backtest_summary.get("window")),
            ("每期组合数", backtest_summary.get("sets")),
            ("模型均值", backtest_summary.get("model_mean")),
            ("模型中位数", backtest_summary.get("model_median")),
            ("模型P90", backtest_summary.get("model_p90")),
            ("随机均值", backtest_summary.get("random_mean")),
            ("随机中位数", backtest_summary.get("random_median")),
            ("随机P90", backtest_summary.get("random_p90")),
        ]
        for klabel, v in kvs:
            md.append(f"- {klabel}: {v}")

    md.append("\n---\n")
    md.append("数据来源： 中国福利彩票官网 findDrawNotice 接口 (name=kl8)\n")
    md.append("项目用途：技术学习与数据分析研究\n")

    # 附：若存在可视化文件，给出相对路径提示
    md.append("\n## 可视化文件\n")
    for p in ["kl8_frequency_hist.png", "kl8_ema_heatmap.png", "kl8_overlap_compare.png", "kl8_dual_frequency_style.png"]:
        pp = os.path.join("plots", p)
        if os.path.exists(os.path.join(PLOTS_DIR, p)):
            md.append(f"- {pp}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md))


# ----------------------------- 可视化 ----------------------------- #

# 辅助函数：柱状图标注
def _annotate_bar(ax, rects, fmt="{:d}", fontsize=9, offset=3):
    """安全标注柱子：
    - 高柱：数值放在柱内顶端（白色）；
    - 矮柱：数值放在柱顶之上，但钳制在坐标轴内，clip_on=True 防止溢出。
    """
    y_min, y_max = ax.get_ylim()
    margin = max(0.06 * (y_max - y_min), 1.0)
    for r in rects:
        h = r.get_height()
        x = r.get_x() + r.get_width() / 2.0
        if h >= (y_max - y_min) * 0.25:
            # 柱内标注
            y = max(h - offset, 0.0)
            va = "top"
            color = "white"
        else:
            # 柱外上方，但不超出绘图区
            y = min(h + offset, y_max - margin)
            va = "bottom"
            color = "black"
        ax.text(x, y, fmt.format(int(h)), ha="center", va=va,
                fontsize=fontsize, color=color, clip_on=True)

def plot_dual_frequency_style(entries: List[dict], k: int = 80, split: int = 40, out_path: str | None = None) -> str:
    """
    生成与“红/蓝条形图上下排布”的风格相近的图：
    - 上面：1..split 的出现频率（红色条）
    - 下面：split+1..k 的出现频率（蓝色条）
    仅为视觉风格适配，快乐8并无红蓝球之分。
    """
    freq = frequency_stats(entries, k=k)
    xs1 = list(range(1, split + 1))
    ys1 = [freq[i] for i in xs1]
    xs2 = list(range(split + 1, k + 1))
    ys2 = [freq[i] for i in xs2]

    fig = plt.figure(figsize=(18, 10), dpi=180)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    rects1 = ax1.bar(xs1, ys1, color="#FF5A5F", edgecolor="#C64245", linewidth=0.4)
    ax1.set_title(f"号码出现频率分布（1 – {split}）", fontsize=18, pad=8)
    ax1.set_xlabel("号码", fontsize=12)
    ax1.set_ylabel("出现次数", fontsize=12)
    ax1.set_xticks(xs1)  # 每 1 个刻度
    ax1.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax1.tick_params(axis="y", labelsize=10)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    y1max = max(ys1) if ys1 else 1
    ax1.set_ylim(0, y1max * 1.15)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    _annotate_bar(ax1, rects1, fmt="{:d}", fontsize=9, offset=3)

    ax2 = fig.add_subplot(gs[1, 0])
    rects2 = ax2.bar(xs2, ys2, color="#4D79FF", edgecolor="#3656B3", linewidth=0.4)
    ax2.set_title(f"号码出现频率分布（{split + 1} – {k}）", fontsize=18, pad=8)
    ax2.set_xlabel("号码", fontsize=12)
    ax2.set_ylabel("出现次数", fontsize=12)
    ax2.set_xticks(xs2)  # 每 1 个刻度
    ax2.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax2.tick_params(axis="y", labelsize=10)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    y2max = max(ys2) if ys2 else 1
    ax2.set_ylim(0, y2max * 1.15)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    _annotate_bar(ax2, rects2, fmt="{:d}", fontsize=9, offset=3)

    out_path = out_path or DUAL_FREQ_FILE
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_frequency_hist(entries: List[dict], k: int = 80, out_path: str | None = None) -> str:
    """号码频率直方图。"""
    freq = frequency_stats(entries, k=k)
    xs = list(range(1, k + 1))
    ys = [freq[i] for i in xs]
    fig = plt.figure(figsize=(18, 8), dpi=180)
    ax = fig.add_subplot(111)

    rects = ax.bar(xs, ys, color="#2F78FF", edgecolor="#234FAD", linewidth=0.4)

    ax.set_title("KL8 号码出现频率（全区 1 – 80）", fontsize=18, pad=10)
    ax.set_xlabel("号码", fontsize=12)
    ax.set_ylabel("出现次数", fontsize=12)

    # 横轴：每 1 个号码一个刻度
    ax.set_xticks(xs)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=10)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 留足顶部空间，避免标注撞顶
    ymax = max(ys) if ys else 1
    ax.set_ylim(0, ymax * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _annotate_bar(ax, rects, fmt="{:d}", fontsize=9, offset=3)

    out_path = out_path or os.path.join(PLOTS_DIR, "kl8_frequency_hist.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path

def plot_ema_heatmap(entries: List[dict], alpha: float = 0.25, k: int = 80, out_path: str | None = None) -> str:
    """EMA 热度热力图（行=时间旧->新，列=号码1..k）。"""
    occ = build_occ_matrix(entries, k=k)
    ema = ema_matrix(occ, alpha=alpha)
    if not ema:
        out_path = out_path or os.path.join(PLOTS_DIR, "kl8_ema_heatmap.png")
        plt.figure()
        plt.savefig(out_path)
        plt.close()
        return out_path
    fig = plt.figure(figsize=(14, 7), dpi=180)
    ax = fig.add_subplot(111)
    im = ax.imshow(ema, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("EMA 强度", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel("号码（1..80）", fontsize=12)
    ax.set_ylabel("时间（旧 → 新）", fontsize=12)
    ax.set_title(f"KL8 EMA 热力图（alpha={alpha}）", fontsize=16, pad=10)
    ax.set_xticks(list(range(0, k, 5)))
    ax.set_xticklabels([str(i if i != 0 else 1) for i in range(0, k, 5)], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    out_path = out_path or os.path.join(PLOTS_DIR, "kl8_ema_heatmap.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
# ----------------------------- 玩法赔率 & 期望收益 ----------------------------- #
def load_payouts(path: str | None) -> dict | None:
    """读取玩法赔率表（JSON）。格式:
    {
      "choose": 7,                # 任选N
      "price_per_bet": 2.0,       # 单注价格
      "payouts": { "0":0, "3":10, "4":28, "5":288, "6":10000, "7":80000 }
    }
    """
    p = path or DEFAULT_PAYOUTS_FILE
    if not os.path.exists(p):
        # 写一个模板，提示用户填写
        template = {
            "choose": 7,
            "price_per_bet": 2.0,
            "payouts": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0}
        }
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def hypergeom_pmf_hits(n_choose: int, h: int, N_total: int = 80, draw: int = 20) -> float:
    """P(命中h) = C(n, h) * C(N-n, draw-h) / C(N, draw)"""
    if h < 0 or h > n_choose or h > draw:
        return 0.0
    num = comb(n_choose, h) * comb(N_total - n_choose, draw - h)
    den = comb(N_total, draw)
    return float(num) / float(den)

def expected_value_for_pick(n_choose: int, payouts: dict, price_per_bet: float) -> float:
    """给定任选N的赔率表，计算单注期望收益（不依赖具体号码，只依赖n）。"""
    ev = 0.0
    for h_str, prize in payouts.items():
        h = int(h_str)
        ev += hypergeom_pmf_hits(n_choose, h) * float(prize)
    return ev - price_per_bet

def kelly_fraction_multi(n_choose: int, payouts: dict, price_per_bet: float, resolution: int = 500) -> float:
    """
    对多结果分布，枚举求解 f∈[0,1] 最大化 E[log(1 + f*(g-1))] 的 Kelly 仓位。
    g = 奖金/单注价格。若 EV<=0，则返回 0。
    """
    outcomes = []  # (p, g)
    ev = 0.0
    for h_str, prize in payouts.items():
        h = int(h_str)
        p = hypergeom_pmf_hits(n_choose, h)
        g = float(prize) / float(price_per_bet)
        outcomes.append((p, g))
        ev += p * (g - 1.0)
    if ev <= 0:
        return 0.0

    best_f, best_obj = 0.0, -1e18
    for i in range(resolution + 1):
        f = i / float(resolution)
        s = 0.0
        valid = True
        for p, g in outcomes:
            r = 1.0 + f * (g - 1.0)
            if r <= 0:
                valid = False
                break
            s += p * math.log(r)
        if not valid:
            continue
        if s > best_obj:
            best_obj, best_f = s, f
    return best_f

def recommend_pickn_sets(entries: List[dict], n: int, sets: int = 5, rng_seed: int | None = None) -> List[List[int]]:
    """与 recommend_sets 一致的加权思路，但只选 n 个号码（适配任选N玩法）。"""
    k = 80
    rng = random.Random(rng_seed)
    # 融合权重
    freq = frequency_stats(entries, k=k)
    freq_weights = [freq[i] + 1e-6 for i in range(1, k + 1)]
    s = float(sum(freq_weights))
    freq_weights = [x / s for x in freq_weights]
    trend_w = trend_weights(entries, k=k, alpha=0.25)
    weights = [0.6 * tw + 0.4 * fw for tw, fw in zip(trend_w, freq_weights)]
    recs = []
    attempts = 0
    while len(recs) < sets and attempts < sets * 30:
        attempts += 1
        pick = sample_without_replacement(weights, n, rng)
        odd = sum(1 for x in pick if x % 2 == 1)
        even = n - odd
        if abs(odd - even) > max(2, n // 3):
            continue
        recs.append(pick)
    return recs

def allocate_budget_by_ev(
    recs: List[List[int]],
    n_choose: int,
    payouts: dict,
    total_budget: float,
    price_per_bet: float,
    weights: List[float] | None = None,
    beta: float = 3.0,
    min_stake: int = 1,
    max_share: float = 0.5,
    kelly_mode: str = "strict",         # 'strict'|'floor'|'ignore'
    kelly_floor_stakes: int = 0,         # floor 模式下的最少下注注数
    fraction_of_kelly: float = 1.0,      # 采用 Kelly 的比例(0~1)
) -> List[dict]:
    """基于 *单注EV + 号码权重分数* 的分散化预算分配。

    - 先计算该玩法的 *单注EV*（与具体选号无关）。
    - 用 *权重得分*（由 `weights` 计算）区分不同组合，softmax 得到分配概率，避免“赢家通吃”。
    - 约束：每组至少 `min_stake` 注（若 EV<=0 则允许为0），单组不超过 `max_share` 的总注数。
    - 返回每项包含 `share`（占比）、`prob`（softmax概率）、`score`（权重分数）。
    """
    if not recs:
        return []
    if total_budget <= 0 or price_per_bet <= 0:
        return []

    # 玩法层面的单注期望收益（不依赖具体号码）
    ev_single = expected_value_for_pick(n_choose, payouts, price_per_bet)

    # Kelly 最优仓位：以本期预算视作 bankroll
    bankroll_stakes = int(total_budget // price_per_bet)
    kelly_f = kelly_fraction_multi(n_choose, payouts, price_per_bet)
    kelly_f = max(0.0, min(1.0, kelly_f * float(fraction_of_kelly)))

    if kelly_mode == "ignore":
        # 忽略 Kelly，使用全部预算
        stakes_total = bankroll_stakes
    elif kelly_mode == "floor":
        # 至少下注若干注，与 Kelly 取较大者
        stakes_total = max(int(kelly_floor_stakes), int(round(kelly_f * bankroll_stakes)))
        stakes_total = min(stakes_total, bankroll_stakes)
    else:  # strict
        stakes_total = int(round(kelly_f * bankroll_stakes))

    if stakes_total <= 0:
        # 返回仅含信息的计划（0 注），提醒负EV/或 Kelly 给 0
        plan = []
        for idx, r in enumerate(recs, 1):
            sc = 1.0 if (weights is None or not weights) else max(pick_weight_score(r, weights), 1e-12)
            plan.append({
                "index": idx, "pick": r,
                "ev_per_bet": round(ev_single, 6),
                "score": round(sc, 6),
                "prob": round(1.0/len(recs), 6),
                "share": 0.0, "stakes": 0, "amount": 0.0
            })
        return plan
        
    # 计算每组的权重分数；若未提供权重，则使用均匀分数
    if weights is None or not weights:
        scores = [1.0 for _ in recs]
    else:
        scores = [max(pick_weight_score(r, weights), 1e-12) for r in recs]

    # softmax 概率（数值稳定）
    import math as _m
    mx = max(scores)
    exps = [_m.exp(beta * (s - mx)) for s in scores]
    Z = sum(exps) if exps else 1.0
    probs = [x / Z for x in exps]

    # 当 EV<=0：允许最小注为0；当 EV>0：默认至少1注以分散风险
    this_min_stake = min_stake if ev_single > 0 else 0

    # 先给每组一个最小注（如果需要），剩余按概率分配
    stakes = [this_min_stake for _ in recs]
    remaining = stakes_total - sum(stakes)
    if remaining < 0:
        # 预算太小，放弃最小注约束，纯按概率分配
        stakes = [0] * len(recs)
        remaining = stakes_total

    # 理想分配（浮点），再做最大占比限制
    ideal = [p * remaining for p in probs]
    base = [int(_m.floor(x)) for x in ideal]
    rem = remaining - sum(base)
    # 用最大小数部分法补齐
    frac_order = sorted(range(len(recs)), key=lambda i: (ideal[i] - base[i]), reverse=True)
    for i in range(rem):
        base[frac_order[i % len(recs)]] += 1

    cap = int(max_share * stakes_total)
    for i in range(len(recs)):
        stakes[i] += base[i]
        if stakes[i] > cap:
            stakes[i] = cap

    # 若未用完预算，按概率在未触顶的组中继续分配
    used = sum(stakes)
    while used < stakes_total:
        candidates = [i for i in range(len(recs)) if stakes[i] < cap]
        if not candidates:
            break
        candidates.sort(key=lambda i: probs[i], reverse=True)
        for i in candidates:
            if used >= stakes_total:
                break
            stakes[i] += 1
            used += 1

    # 出具计划
    plan = []
    for idx, r in enumerate(recs, 1):
        st = int(stakes[idx - 1])
        amount = round(st * price_per_bet, 2)
        share = (st / stakes_total) if stakes_total > 0 else 0.0
        plan.append({
            "index": idx,
            "pick": r,
            "ev_per_bet": round(ev_single, 6),
            "score": round(scores[idx - 1], 6),
            "prob": round(probs[idx - 1], 6),
            "share": round(share, 4),
            "stakes": st,
            "amount": amount,
        })
    return plan

def plot_overlap_hist_compare(model_overlaps: List[int], random_overlaps: List[int], out_path: str | None = None) -> str:
    """模型 vs 随机 的命中重叠直方图对比。"""
    plt.figure()
    # 统一的 bins
    bins = range(0, 21)  # 0..20
    plt.hist(model_overlaps, bins=bins, alpha=0.5, label="Model")
    plt.hist(random_overlaps, bins=bins, alpha=0.5, label="Random")
    plt.xlabel("Overlap with Actual (0..20)")
    plt.ylabel("Count")
    plt.title("Model vs Random Overlap Distribution")
    plt.legend()
    out_path = out_path or os.path.join(PLOTS_DIR, "kl8_overlap_compare.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


# ----------------------------- CLI ----------------------------- #

def main():
    parser = argparse.ArgumentParser(description="快乐8 数据抓取与分析")
    parser.add_argument("--fetch", action="store_true", help="抓取并保存历史数据")
    parser.add_argument("--limit", type=int, default=0, help="最近 N 期（0 表示接口默认分页）")
    parser.add_argument("--recommend", type=int, default=0, help="输出推荐组数（示例用途）")
    parser.add_argument("--report", action="store_true", help="生成 Markdown 报告")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现推荐）")
    parser.add_argument("--backtest", action="store_true", help="执行滚动回测并与随机基线对比")
    parser.add_argument("--window", type=int, default=200, help="回测窗口期数")
    parser.add_argument("--sets_per_period", type=int, default=5, help="每期推荐的组合数（用于回测）")
    parser.add_argument("--random_trials", type=int, default=2000, help="随机基线样本量（总计近似）")
    parser.add_argument("--plots", action="store_true", help="生成频率直方图与EMA热力图")
    parser.add_argument("--plots_dual", action="store_true", help="生成上下分区红/蓝风格的频率分布图（视觉风格适配）")
    parser.add_argument("--split", type=int, default=40, help="双面板分割点（默认40，将1..split放上面）")
    parser.add_argument("--plan", action="store_true", help="根据权重分数与预算生成资金分配方案（学习用途）")
    parser.add_argument("--budget", type=float, default=22.0, help="总预算金额（元）")
    parser.add_argument("--price_per_bet", type=float, default=2.0, help="单注价格（元）")
    parser.add_argument("--play_pick", type=int, default=0, help="任选N（如 7 表示选七玩法）；>0 时按该玩法推荐与分配")
    parser.add_argument("--payouts", type=str, default=None, help="玩法赔率JSON路径（默认 data/payouts_kl8.json）")
    parser.add_argument("--beta", type=float, default=3.0, help="Softmax温度，越大分配越集中")
    parser.add_argument("--max_share", type=float, default=0.5, help="单组最大注数占比(0~1)")
    parser.add_argument("--min_stake", type=int, default=1, help="EV>0时每组最小注数")
    parser.add_argument("--kelly_mode", choices=["strict", "floor", "ignore"], default="strict",
                    help="Kelly 模式：strict=遵循Kelly；floor=至少下注若干注；ignore=忽略Kelly用全部预算")
    parser.add_argument("--kelly_floor_stakes", type=int, default=0,
                        help="kelly_mode=floor 时的最少下注注数")
    parser.add_argument("--fraction_of_kelly", type=float, default=1.0,
                        help="采用 Kelly 的比例(0~1)，如 0.5 表示半Kelly")
    args = parser.parse_args()

    # 读取已存数据
    history = load_existing()

    if args.fetch:
        print("[INFO] 正在抓取快乐8数据...")
        fetched = fetch_kl8_history(limit=(args.limit or None))
        normed = [normalize_entry(e) for e in fetched if e]
        # 简单去重: 以期号为键
        exist_codes = {h.get("code") for h in history}
        merged = history + [e for e in normed if e.get("code") not in exist_codes]
        # 按期号/日期排序（期号非严格可排序，优先按日期）
        merged.sort(key=lambda x: (x.get("date", ""), x.get("code", "")), reverse=True)
        if args.limit and args.limit > 0:
            merged = merged[:args.limit]
            print(f"[INFO] 按 --limit={args.limit} 截断为最近 {len(merged)} 期")
        save_history(merged)
        history = merged
        print(f"[INFO] 已保存 {len(history)} 期到 {DATA_FILE}")

    if not history:
        print("[WARN] 无本地数据，可加 --fetch 抓取。")

    # 可视化
    if args.plots and history:
        freq_png = plot_frequency_hist(history)
        ema_png = plot_ema_heatmap(history, alpha=0.25)
        print(f"[INFO] 已生成可视化: {freq_png} ; {ema_png}")

    if args.plots_dual and history:
        dual_png = plot_dual_frequency_style(history, k=80, split=args.split)
        print(f"[INFO] 已生成双面板频率图: {dual_png}")

    bt = None
    if args.backtest and history:
        bt = backtest_overlap(history, window=args.window, sets=args.sets_per_period, alpha=0.25, rng_seed=(args.seed or 123), random_trials=args.random_trials)
        print("[INFO] 回测摘要:", bt["summary"])
        # 画对比图
        cmp_png = plot_overlap_hist_compare(bt["model_overlaps"], bt["random_overlaps"])
        print(f"[INFO] 已生成回测对比图: {cmp_png}")

    recs: List[List[int]] = []
    if args.recommend and history:
        if args.play_pick and args.play_pick > 0:
            recs = recommend_pickn_sets(history, n=args.play_pick, sets=args.recommend, rng_seed=args.seed)
        else:
            recs = recommend_sets(history, sets=args.recommend, rng_seed=args.seed)
        for i, r in enumerate(recs, 1):
            print(f"方案{i}: ", ", ".join(f"{x:02d}" for x in r))

    # 资金分配方案
    plan = None
    if args.plan and history:
        if args.play_pick and args.play_pick > 0:
            payouts_cfg = load_payouts(args.payouts)
            if payouts_cfg and int(payouts_cfg.get("choose", args.play_pick)) == args.play_pick:
                price = float(payouts_cfg.get("price_per_bet", args.price_per_bet))
                weights_for_plan = trend_weights(history, k=80, alpha=0.25)
                plan = allocate_budget_by_ev(
                    recs if recs else recommend_pickn_sets(history, n=args.play_pick, sets=5, rng_seed=args.seed),
                    n_choose=args.play_pick,
                    payouts=payouts_cfg.get("payouts", {}),
                    total_budget=args.budget,
                    price_per_bet=price,
                    weights=weights_for_plan,
                    beta=args.beta,
                    min_stake=args.min_stake,
                    max_share=args.max_share,
                    kelly_mode=args.kelly_mode,
                    kelly_floor_stakes=args.kelly_floor_stakes,
                    fraction_of_kelly=args.fraction_of_kelly,
                )
                plan_path = write_profit_plan(history, recs if recs else [], plan, args.budget, price)
                print(f"[INFO] 已生成资金分配方案(基于EV): {plan_path}")
            else:
                # 无赔率表，退回权重分配
                weights_for_plan = trend_weights(history, k=80, alpha=0.25)
                plan = allocate_budget_by_score(recs if recs else recommend_pickn_sets(history, n=args.play_pick, sets=5, rng_seed=args.seed),
                                                weights_for_plan, total_budget=args.budget, price_per_bet=args.price_per_bet)
                plan_path = write_profit_plan(history, recs if recs else [], plan, args.budget, args.price_per_bet)
                print(f"[WARN] 未找到有效赔率表，已按权重分配: {plan_path}")
        else:
            weights_for_plan = trend_weights(history, k=80, alpha=0.25)
            plan = allocate_budget_by_score(recs if recs else recommend_sets(history, sets=5, rng_seed=args.seed),
                                            weights_for_plan, total_budget=args.budget, price_per_bet=args.price_per_bet)
            plan_path = write_profit_plan(history, recs if recs else [], plan, args.budget, args.price_per_bet)
            print(f"[INFO] 已生成资金分配方案: {plan_path}")

    if args.report and history:
        write_report(history, recs, backtest_summary=(bt["summary"] if bt else None))
        print(f"[INFO] 已生成报告: {REPORT_FILE}")


if __name__ == "__main__":
    main()