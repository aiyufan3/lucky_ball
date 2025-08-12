#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球开奖数据抓取与分析脚本

⚠️  重要免责声明 ⚠️
1. 本脚本仅用于技术学习和数据分析研究目的
2. 彩票开奖结果完全随机，历史数据无法预测未来结果
3. 本分析结果仅供参考，不构成任何投注建议
4. 请理性购彩，量力而行，未满18周岁禁止购买彩票
5. 开发者不承担因使用本脚本产生的任何损失

功能：
1. 抓取中国福利彩票双色球历史开奖数据
2. 分析开奖号码规律
3. 基于统计分析生成推荐号码
"""

import requests
import time
import json
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import warnings
import os
import hjson
import random
import torch
import sys
import torch.nn as nn
import torch.optim as optim
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import entropy as scipy_entropy
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DoubleColorBallAnalyzer:
    """双色球分析器"""
    
    def __init__(self):
        self.base_url = "https://www.cwl.gov.cn/ygkj/wqkjgg/"
        self.api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
        
        # 多个真实的User-Agent，用于轮换
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0'
        ]
    
        self.session = requests.Session()
        self.lottery_data = []

        # ML related
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.red_model = None      # LSTM for red (multi-label)
        self.blue_model = None     # LSTM for blue (single-label)
        self.seq_len = 10
        self.trained = False

        # Reproducibility
        self.seed = 2025
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # 配置session
        self._setup_session()
        
    def _setup_session(self):
        """配置session的基本设置"""
        # 设置连接池
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # 设置基本headers
        self._update_headers()
    
    def _update_headers(self):
        """更新请求头，使用随机User-Agent"""
        user_agent = random.choice(self.user_agents)
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        self.session.headers.update(headers)
        print(f"🔄 更新User-Agent: {user_agent[:50]}...")
        
    def get_max_pages(self):
        """获取真实的最大页码，增强错误处理"""
        print("正在获取最大页码...")
        
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # 每次尝试都更新headers
                self._update_headers()
                
                # 先获取第一页数据来确定总数
                params = {
                    'name': 'ssq',
                    'pageNo': 1,
                    'pageSize': 30,
                    'systemType': 'PC'
                }
                
                # 添加随机延时
                # if attempt > 0:
                #     delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                #     print(f"⏳ 第 {attempt + 1} 次尝试，等待 {delay:.1f} 秒...")
                #     time.sleep(delay)
                # else:
                #     time.sleep(random.uniform(0, 1))  # 初始随机延时
                
                print(f"🌐 正在请求API... (尝试 {attempt + 1}/{max_retries})")
                response = self.session.get(self.api_url, params=params, timeout=30)
                
                print(f"📡 响应状态码: {response.status_code}")
                response.raise_for_status()
                
                data = response.json()
                print(f"📊 API响应: state={data.get('state')}, message={data.get('message')}")
                
                if data.get('state') != 0:
                    print(f"❌ API返回错误: {data.get('message', '未知错误')}")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return 100  # 默认返回100页
                
                # 尝试获取总记录数
                total_count = data.get('total', 0)
                if total_count > 0:
                    max_pages = (total_count + 29) // 30  # 向上取整
                    print(f"✅ 发现总共 {total_count} 条记录，需要抓取 {max_pages} 页")
                    return max_pages
                
                # 如果无法获取总数，通过试探方式确定最大页码
                print("无法获取总记录数，正在试探最大页码...")
                page = 1
                while page <= 200:  # 设置上限防止无限循环
                    params['pageNo'] = page
                    response = self.session.get(self.api_url, params=params, timeout=30)
                    data = response.json()
                    
                    if data.get('state') != 0 or not data.get('result'):
                        break
                    
                    page += 10  # 每次跳跃10页快速试探
                    time.sleep(0.2)
                
                # 精确定位最大页码
                start = max(1, page - 10)
                end = page
                
                for precise_page in range(start, end + 1):
                    params['pageNo'] = precise_page
                    response = self.session.get(self.api_url, params=params, timeout=30)
                    data = response.json()
                    
                    if data.get('state') != 0 or not data.get('result'):
                        max_pages = precise_page - 1
                        print(f"✅ 通过试探确定最大页码为 {max_pages}")
                        return max_pages
                    
                    time.sleep(0.1)
                
                return max(1, page - 1)
                
            except requests.exceptions.Timeout:
                print(f"⏰ 请求超时 (尝试 {attempt + 1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                print(f"🔌 连接错误 (尝试 {attempt + 1}/{max_retries})")
            except requests.exceptions.HTTPError as e:
                print(f"🌐 HTTP错误: {e} (尝试 {attempt + 1}/{max_retries})")
            except Exception as e:
                print(f"❌ 获取最大页码时出错: {e} (尝试 {attempt + 1}/{max_retries})")
            
            if attempt < max_retries - 1:
                print("🔄 准备重试...")
        
        print("⚠️  所有尝试都失败，使用默认页数 100")
        return 100
    
    def fetch_lottery_data(self, max_pages=10):
        """抓取双色球开奖数据，增强错误处理"""
        print("开始抓取双色球开奖数据...")
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        successful_pages = 0
        
        for page in range(1, max_pages + 1):
            print(f"📄 正在抓取第 {page} 页数据...")
            
            # 重试机制
            max_retries = 5
            retry_count = 0
            success = False
            base_delay = 1
            
            while retry_count < max_retries and not success:
                try:
                    # 每隔几次请求更新headers
                    if page % 5 == 1 or retry_count > 0:
                        self._update_headers()
                    
                    # API参数
                    params = {
                        'name': 'ssq',  # 双色球
                        'pageNo': page,
                        'pageSize': 30,
                        'systemType': 'PC'
                    }
                    
                    # 计算延时
                    # if retry_count > 0:
                    #     # 指数退避延时，加上随机因子
                    #     delay = min(base_delay * (2 ** retry_count), 15) + random.uniform(0, 2)
                    #     print(f"⏳ 第 {retry_count + 1} 次重试，等待 {delay:.1f} 秒...")
                    #     time.sleep(delay)
                    # else:
                    #     # 正常延时，避免请求过于频繁
                    #     delay = random.uniform(1, 2)  # 1-2秒随机延时
                    #     time.sleep(delay)
                    
                    print(f"🌐 发送请求到API... (页面 {page}, 尝试 {retry_count + 1})")
                    response = self.session.get(self.api_url, params=params, timeout=30)
                    
                    print(f"📡 响应状态码: {response.status_code}")
                    response.raise_for_status()
                    
                    # 解析JSON响应
                    data = response.json()
                    print(f"📊 API响应解析: state={data.get('state')}")
                    
                    if data.get('state') != 0:
                        print(f"❌ API返回错误: {data.get('message', '未知错误')}")
                        retry_count += 1
                        continue
                    
                    results = data.get('result', [])
                    if not results:
                        print(f"📭 第 {page} 页无数据")
                        break
                    
                    print(f"✅ 第 {page} 页获取到 {len(results)} 条记录")
                    consecutive_failures = 0  # 重置连续失败计数
                    successful_pages += 1
                    
                    for item in results:
                        try:
                            # 解析期号
                            period = item.get('code', '')
                            
                            # 解析开奖日期
                            date_str = item.get('date', '')
                            # 提取日期部分，去除星期信息
                            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_str)
                            if not date_match:
                                continue
                            draw_date = date_match.group(1)
                            
                            # 解析红球号码（逗号分隔的字符串）
                            red_str = item.get('red', '')
                            if not red_str:
                                continue
                            red_balls = [int(x.strip()) for x in red_str.split(',')]
                            
                            # 解析蓝球号码
                            blue_str = item.get('blue', '')
                            if not blue_str:
                                continue
                            blue_ball = int(blue_str)
                            
                            # 解析其他信息
                            sales_amount = self._parse_number(item.get('sales', '0'))
                            pool_amount = self._parse_number(item.get('poolmoney', '0'))
                            
                            # 解析奖级信息
                            prizegrades = item.get('prizegrades', [])
                            first_prize_count = 0
                            first_prize_amount = 0
                            second_prize_count = 0
                            second_prize_amount = 0
                            
                            for grade in prizegrades:
                                if grade.get('type') == 1:  # 一等奖
                                    first_prize_count = self._parse_number(grade.get('typenum', '0'))
                                    first_prize_amount = self._parse_number(grade.get('typemoney', '0'))
                                elif grade.get('type') == 2:  # 二等奖
                                    second_prize_count = self._parse_number(grade.get('typenum', '0'))
                                    second_prize_amount = self._parse_number(grade.get('typemoney', '0'))
                            
                            # 存储数据
                            lottery_record = {
                                'period': period,
                                'date': draw_date,
                                'red_balls': red_balls,
                                'blue_ball': blue_ball,
                                'first_prize_count': first_prize_count,
                                'first_prize_amount': first_prize_amount,
                                'second_prize_count': second_prize_count,
                                'second_prize_amount': second_prize_amount,
                                'sales_amount': sales_amount,
                                'pool_amount': pool_amount
                            }
                            
                            self.lottery_data.append(lottery_record)
                            
                        except Exception as e:
                            print(f"⚠️  解析记录时出错: {e}")
                            continue
                    
                    success = True  # 标记成功
                    
                except requests.exceptions.Timeout:
                    print(f"⏰ 网络超时 (页面 {page}, 尝试 {retry_count + 1})")
                    retry_count += 1
                except requests.exceptions.ConnectionError:
                    print(f"🔌 连接错误 (页面 {page}, 尝试 {retry_count + 1})")
                    retry_count += 1
                except requests.exceptions.HTTPError as e:
                    print(f"🌐 HTTP错误: {e} (页面 {page}, 尝试 {retry_count + 1})")
                    retry_count += 1
                except Exception as e:
                    print(f"❌ 抓取第 {page} 页时出错: {e} (尝试 {retry_count + 1})")
                    retry_count += 1
                
                if retry_count >= max_retries:
                    consecutive_failures += 1
                    print(f"💥 第 {page} 页重试 {max_retries} 次后仍然失败，跳过此页")
                    break
            
            # 如果连续失败太多次，停止抓取
            if consecutive_failures >= max_consecutive_failures:
                print(f"🛑 连续 {max_consecutive_failures} 页失败，停止抓取以避免被封禁")
                break
        
        print(f"🎉 数据抓取完成！成功抓取 {successful_pages} 页，共获取 {len(self.lottery_data)} 期开奖数据")
        
        # 如果获取的数据太少，给出警告
        if len(self.lottery_data) < 100:
            print(f"⚠️  获取的数据较少 ({len(self.lottery_data)} 期)，可能存在网络问题")
        
        return self.lottery_data
    
    def _parse_number(self, text):
        """解析数字，移除逗号等格式符号"""
        if not text or text == '-':
            return 0
        # 移除逗号、￥符号等
        cleaned = re.sub(r'[,￥¥元]', '', str(text))
        try:
            return int(float(cleaned))
        except:
            return 0
    
    def save_data(self, filename="data/lottery_data.json"):
        """保存数据到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.lottery_data, f, ensure_ascii=False, indent=2)
        print(f"数据已保存到 {filename}")
    
    def load_data(self, filename="data/lottery_data.json"):
        """从文件加载数据"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.lottery_data = json.load(f)
            print(f"从 {filename} 加载了 {len(self.lottery_data)} 期数据")
            return True
        except FileNotFoundError:
            print(f"文件 {filename} 不存在")
            return False
    
    def analyze_frequency(self):
        """分析号码出现频率"""
        print("\n=== 号码频率分析 ===")
        
        # 红球频率分析
        red_counter = Counter()
        blue_counter = Counter()
        
        for record in self.lottery_data:
            for red in record['red_balls']:
                red_counter[red] += 1
            blue_counter[record['blue_ball']] += 1
        
        # 红球频率排序
        red_freq = sorted(red_counter.items(), key=lambda x: x[1], reverse=True)
        print("\n红球出现频率排行榜（前10）：")
        for i, (num, count) in enumerate(red_freq[:10], 1):
            percentage = (count / len(self.lottery_data)) * 100
            print(f"{i:2d}. 号码 {num:2d}: 出现 {count:3d} 次 ({percentage:.1f}%)")
        
        # 蓝球频率排序
        blue_freq = sorted(blue_counter.items(), key=lambda x: x[1], reverse=True)
        print("\n蓝球出现频率排行榜（前10）：")
        for i, (num, count) in enumerate(blue_freq[:10], 1):
            percentage = (count / len(self.lottery_data)) * 100
            print(f"{i:2d}. 号码 {num:2d}: 出现 {count:3d} 次 ({percentage:.1f}%)")
        
        return red_counter, blue_counter
    
    # -------------------- ML helpers & models --------------------
    class _SeqPredictor(nn.Module):
        """Generic LSTM predictor that emits a single-step prediction."""
        def __init__(self, input_size, hidden_size, output_size, output_type="sigmoid", dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
            self.dropout = nn.Dropout(p=dropout)
            self.head = nn.Linear(hidden_size, output_size)
            self.output_type = output_type

        def forward(self, x):
            # x: (B, T, F)
            out, _ = self.lstm(x)
            last = out[:, -1, :]
            last = self.dropout(last)
            logits = self.head(last)
            if self.output_type == "sigmoid":
                return logits, torch.sigmoid(logits)
            elif self.output_type == "softmax":
                return logits, torch.softmax(logits, dim=-1)
            else:
                return logits, logits
            
    def _engineered_features(self, reds):
        """
        Hand-crafted features from a draw:
        - sum (min 21, max 183) -> min-max normalized
        - span (max-min), max 32 -> normalized to [0,1]
        - odd_ratio, even_ratio
        """
        s = sum(reds)
        s_norm = (s - 21.0) / 162.0  # 183-21=162
        span = max(reds) - min(reds)
        span_norm = span / 32.0
        odd = sum(1 for r in reds if r % 2 == 1)
        even = 6 - odd
        return np.array([s_norm, span_norm, odd / 6.0, even / 6.0], dtype=np.float32)
    
    def _onehot_multi(self, reds, blue):
        """
        Convert a draw to multi-hot (33) + one-hot (16) + engineered(4) -> (53,)
        """
        red_vec = np.zeros(33, dtype=np.float32)
        for r in reds:
            if 1 <= r <= 33:
                red_vec[r-1] = 1.0
        blue_vec = np.zeros(16, dtype=np.float32)
        if 1 <= blue <= 16:
            blue_vec[blue-1] = 1.0
        feats = self._engineered_features(reds)
        return np.concatenate([red_vec, blue_vec, feats], axis=0)

    def _build_sequence_dataset(self, seq_len=10):
        """
        Build X (N, seq_len, 53), y_red (N,33), y_blue (N,)
        Newest-first in memory -> sort to oldest-first for sequences.
        """
        if not self.lottery_data or len(self.lottery_data) <= seq_len:
            return None
        data_sorted = sorted(self.lottery_data, key=lambda r: (r["date"], r["period"]))
        feats = [self._onehot_multi(r["red_balls"], r["blue_ball"]) for r in data_sorted]
        X, y_red, y_blue = [], [], []
        for i in range(seq_len, len(feats)):
            X.append(np.stack(feats[i-seq_len:i], axis=0))  # (seq_len,53)
            red_vec = np.zeros(33, dtype=np.float32)
            for rr in data_sorted[i]["red_balls"]:
                red_vec[rr-1] = 1.0
            y_red.append(red_vec)
            y_blue.append(int(data_sorted[i]["blue_ball"]) - 1)
        X = np.stack(X, axis=0).astype(np.float32)
        y_red = np.stack(y_red, axis=0).astype(np.float32)
        y_blue = np.array(y_blue, dtype=np.int64)
        return X, y_red, y_blue
    
    def _sym_kl(self, p, q):
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
        q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        return float(kl_pq + kl_qp)
    
    def time_decay_weights(self, n_rows, half_life=60):
        """Exponential time-decay weights old->new."""
        idx = np.arange(n_rows)
        lam = np.log(2) / max(1, half_life)
        return np.exp(lam * (idx - n_rows + 1))

    def compute_marginal_probs(self, decay_half_life=60):
        """Time-decayed marginal probabilities for red(33) and blue(16)."""
        if not self.lottery_data:
            return np.ones(33)/33.0, np.ones(16)/16.0
        data_sorted = sorted(self.lottery_data, key=lambda r: (r["date"], r["period"]))
        reds_list = [r["red_balls"] for r in data_sorted]
        blues_list = [r["blue_ball"] for r in data_sorted]
        n = len(reds_list)
        w = self.time_decay_weights(n, half_life=decay_half_life)
        red_counts = np.zeros(33, dtype=np.float64)
        blue_counts = np.zeros(16, dtype=np.float64)
        for balls, wb, weight in zip(reds_list, blues_list, w):
            for b in balls:
                red_counts[b-1] += weight
            blue_counts[wb-1] += weight
        pr = red_counts / red_counts.sum() if red_counts.sum() > 0 else np.ones(33)/33.0
        pb = blue_counts / blue_counts.sum() if blue_counts.sum() > 0 else np.ones(16)/16.0
        return pr.astype(np.float32), pb.astype(np.float32)

    def compute_entropy(self, probs):
        """Shannon entropy (bits)."""
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
        return float(scipy_entropy(p, base=2))

    def train_ml_model(self, seq_len=10, epochs=5, lr=1e-3, hidden_size=64, dropout=0.2):
        """
        Train two LSTM predictors:
        - red: multi-label (33 outputs, BCEWithLogitsLoss)
        - blue: single-label (16 outputs, CrossEntropyLoss)
        """
        self.seq_len = seq_len
        ds = self._build_sequence_dataset(seq_len=seq_len)
        if ds is None:
            print("数据不足，跳过模型训练。")
            self.trained = False
            return
        X, y_red, y_blue = ds
        X_t = torch.from_numpy(X).to(self.device)
        y_red_t = torch.from_numpy(y_red).to(self.device)
        y_blue_t = torch.from_numpy(y_blue).to(self.device)

        input_size = X_t.shape[-1]
        self.red_model  = self._SeqPredictor(input_size, hidden_size, 33, output_type="sigmoid", dropout=dropout).to(self.device)
        self.blue_model = self._SeqPredictor(input_size, hidden_size, 16, output_type="softmax",  dropout=dropout).to(self.device)

        red_criterion = nn.BCEWithLogitsLoss()
        blue_criterion = nn.CrossEntropyLoss()
        red_opt = optim.Adam(self.red_model.parameters(), lr=lr)
        blue_opt = optim.Adam(self.blue_model.parameters(), lr=lr)

        batch_size = min(128, X_t.shape[0])
        num_batches = int(np.ceil(X_t.shape[0] / batch_size))

        self.red_model.train(); self.blue_model.train()
        for ep in range(epochs):
            perm = torch.randperm(X_t.shape[0])
            X_t = X_t[perm]; y_red_t = y_red_t[perm]; y_blue_t = y_blue_t[perm]
            red_loss_sum = 0.0; blue_loss_sum = 0.0
            for bi in range(num_batches):
                s = bi * batch_size; e = min((bi+1) * batch_size, X_t.shape[0])
                xb = X_t[s:e]; yrb = y_red_t[s:e]; ybb = y_blue_t[s:e]

                red_opt.zero_grad()
                red_logits, _ = self.red_model(xb)
                loss_red = red_criterion(red_logits, yrb)
                loss_red.backward(); red_opt.step()

                blue_opt.zero_grad()
                blue_logits, _ = self.blue_model(xb)
                loss_blue = blue_criterion(blue_logits, ybb)
                loss_blue.backward(); blue_opt.step()

                red_loss_sum += loss_red.item(); blue_loss_sum += loss_blue.item()
            print(f"[LSTM] epoch {ep+1}/{epochs} red_loss={red_loss_sum/num_batches:.4f} blue_loss={blue_loss_sum/num_batches:.4f}")

        self.trained = True

    def predict_next_probabilities(self, blend_alpha="auto", decay_half_life=60):
        """
        Next-step probabilities for red(33) and blue(16).
        If blend_alpha == "auto": tune alpha by comparing ML vs. marginal distributions
        using symmetric KL divergence (smaller divergence -> larger alpha).
        """
        p_marg_red, p_marg_blue = self.compute_marginal_probs(decay_half_life=decay_half_life)
        if not self.trained:
            print("模型未训练，使用时间衰减频率作为概率。")
            return p_marg_red, p_marg_blue

        ds = self._build_sequence_dataset(seq_len=self.seq_len)
        if ds is None:
            return p_marg_red, p_marg_blue
        X, _, _ = ds
        last_seq = torch.from_numpy(X[-1:]).to(self.device)

        self.red_model.eval(); self.blue_model.eval()
        with torch.no_grad():
            _, p_red_ml = self.red_model(last_seq)   # (1,33)
            _, p_blue_ml = self.blue_model(last_seq) # (1,16)
        p_red_ml = p_red_ml.squeeze(0).cpu().numpy()
        p_blue_ml = p_blue_ml.squeeze(0).cpu().numpy()

        # normalize reds into a distribution
        p_red_ml = p_red_ml / (p_red_ml.sum() + 1e-12)

        # auto-tune alpha
        if blend_alpha == "auto":
            d_red = self._sym_kl(p_red_ml, p_marg_red)
            d_blue = self._sym_kl(p_blue_ml, p_marg_blue)
            d = 0.7 * d_red + 0.3 * d_blue
            alpha = 1.0 / (1.0 + 4.0 * d)  # heuristic scaling
            alpha = float(np.clip(alpha, 0.25, 0.8))
            print(f"🔧 自适应融合系数 alpha={alpha:.3f} (基于分布差异)")
        else:
            alpha = float(blend_alpha)

        p_red = alpha * p_red_ml + (1.0 - alpha) * p_marg_red
        p_blue = alpha * p_blue_ml + (1.0 - alpha) * p_marg_blue
        p_red = p_red / p_red.sum(); p_blue = p_blue / p_blue.sum()
        return p_red.astype(np.float32), p_blue.astype(np.float32)

    def _arima_sum_forecast(self, horizon=1):
            """
            ARIMA on red sums to get (mu, low, high) for next draw.
            """
            try:
                series = [sum(r["red_balls"]) for r in sorted(self.lottery_data, key=lambda r: (r["date"], r["period"]))]
                if len(series) < 30:
                    mu = np.mean(series)
                    return float(mu), float(mu-20), float(mu+20)
                model = ARIMA(series, order=(2,1,2))
                model_fit = model.fit()
                pred = model_fit.get_forecast(steps=horizon)
                # predicted_mean can be ndarray/Series depending on input; coerce to ndarray
                mu = float(np.asanyarray(pred.predicted_mean)[-1])
                ci = pred.conf_int(alpha=0.20)  # 80%
                try:
                    # pandas DataFrame case
                    low = float(ci.iloc[-1, 0]); high = float(ci.iloc[-1, 1])
                except AttributeError:
                    # numpy ndarray case: shape (steps, 2)
                    low = float(ci[-1, 0]); high = float(ci[-1, 1])
                return mu, low, high
            except Exception as e:
                print(f"ARIMA 预测失败，使用经验范围: {e}")
                series = [sum(r["red_balls"]) for r in self.lottery_data]
                mu = np.mean(series)
                return float(mu), float(mu-25), float(mu+25)

    def _sample_combo_no_replace(self, prob_red, k=6):
        """Sample k distinct red numbers without replacement according to prob_red."""
        prob = prob_red.copy().astype(np.float64)
        prob = prob / prob.sum()
        chosen = []
        available = np.arange(1, 34)
        for _ in range(k):
            idxs = np.array([i-1 for i in available])
            p = prob[idxs]; p = p / p.sum()
            pick = np.random.choice(len(available), p=p)
            val = int(available[pick])
            chosen.append(val)
            available = np.array([x for x in available if x != val])
        return sorted(chosen)

    def _monte_carlo_candidates(self, p_red, p_blue, n=2000, sum_mu=None, sum_low=None, sum_high=None):
            """
            Monte Carlo guided by probabilities & optional sum constraints.
            Returns list of (reds, blue, score, entropy_bits).
            """
            candidates = {}
            for _ in range(n):
                reds = self._sample_combo_no_replace(p_red, k=6)
                s = sum(reds)
                if sum_low is not None and sum_high is not None and not (sum_low <= s <= sum_high):
                    continue
                # entropy over the selected set
                q = np.array([p_red[r-1] for r in reds], dtype=np.float64)
                q = q / q.sum()
                H = self.compute_entropy(q)  # 0..log2(6)
                mean_p = float(np.mean([p_red[r-1] for r in reds]))
                score = 0.7 * mean_p - 0.3 * (H / np.log2(6))
                # adaptive top-k for blue: sharper distribution -> smaller k
                sharp = float(np.max(p_blue))
                k_adapt = int(np.clip(6 - round(4 * sharp / (np.max(p_blue) + 1e-12)), 2, 6))
                k_adapt = min(k_adapt, len(p_blue))
                top_idx = np.argsort(p_blue)[-k_adapt:]
                top_probs = p_blue[top_idx] / p_blue[top_idx].sum()
                blue_idx = int(top_idx[np.random.choice(len(top_idx), p=top_probs)]) + 1
                key = (tuple(reds), blue_idx)
                if key not in candidates or score > candidates[key][0]:
                    candidates[key] = (score, H)
            out = []
            for (reds, blue), (score, H) in candidates.items():
                out.append((list(reds), int(blue), float(score), float(H)))
            out.sort(key=lambda x: -x[2])
            return out
    
    # ------------------ end of ML helpers & models ------------------


    def analyze_patterns(self):
        """分析号码规律"""
        print("\n=== 号码规律分析 ===")
        
        # 奇偶分布分析
        odd_even_dist = defaultdict(int)
        sum_dist = defaultdict(int)
        span_dist = defaultdict(int)
        
        for record in self.lottery_data:
            red_balls = record['red_balls']
            
            # 奇偶分析
            odd_count = sum(1 for x in red_balls if x % 2 == 1)
            even_count = 6 - odd_count
            odd_even_dist[f"{odd_count}奇{even_count}偶"] += 1
            
            # 和值分析
            total_sum = sum(red_balls)
            sum_range = f"{(total_sum//10)*10}-{(total_sum//10)*10+9}"
            sum_dist[sum_range] += 1
            
            # 跨度分析
            span = max(red_balls) - min(red_balls)
            span_range = f"{(span//5)*5}-{(span//5)*5+4}"
            span_dist[span_range] += 1
        
        print("\n奇偶分布统计：")
        for pattern, count in sorted(odd_even_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.lottery_data)) * 100
            print(f"{pattern}: {count} 次 ({percentage:.1f}%)")
        
        print("\n和值分布统计：")
        for sum_range, count in sorted(sum_dist.items(), key=lambda x: int(x[0].split('-')[0])):
            percentage = (count / len(self.lottery_data)) * 100
            print(f"{sum_range}: {count} 次 ({percentage:.1f}%)")
        
        print("\n跨度分布统计：")
        for span_range, count in sorted(span_dist.items(), key=lambda x: int(x[0].split('-')[0])):
            percentage = (count / len(self.lottery_data)) * 100
            print(f"{span_range}: {count} 次 ({percentage:.1f}%)")
    
    def analyze_trends(self):
        """分析走势"""
        print("\n=== 走势分析 ===")
        
        if len(self.lottery_data) < 10:
            print("数据不足，无法进行走势分析")
            return
        
        # 最近10期的号码（按日期倒序）
        data_sorted = sorted(self.lottery_data, key=lambda r: (r['date'], r['period']), reverse=True)
        recent_10 = data_sorted[:10]
        
        print("最近10期开奖号码：")
        for record in recent_10:
            red_str = " ".join([f"{x:2d}" for x in record['red_balls']])
            print(f"{record['period']}: {red_str} + {record['blue_ball']:2d}")
        
        # 冷热号分析
        red_counter = Counter()
        blue_counter = Counter()
        
        for record in recent_10:
            for red in record['red_balls']:
                red_counter[red] += 1
            blue_counter[record['blue_ball']] += 1
        
        print(f"\n最近10期红球热号（出现2次及以上）：")
        hot_reds = [num for num, count in red_counter.items() if count >= 2]
        if hot_reds:
            hot_reds.sort()
            print(" ".join([f"{x:2d}" for x in hot_reds]))
        else:
            print("无")
        
        print(f"\n最近10期蓝球热号（出现2次及以上）：")
        hot_blues = [num for num, count in blue_counter.items() if count >= 2]
        if hot_blues:
            hot_blues.sort()
            print(" ".join([f"{x:2d}" for x in hot_blues]))
        else:
            print("无")
    
    def generate_recommendations(self, num_sets=5):
        """使用 LSTM + 时间衰减频率 + ARIMA 约束 + Monte Carlo 的低熵推荐"""
        print(f"\n=== 生成 {num_sets} 组机器学习增强推荐 ===")
        if not self.lottery_data:
            print("无数据，无法生成推荐")
            return []

        # 1) fused probabilities (ML + time-decayed marginals)
        p_red, p_blue = self.predict_next_probabilities(blend_alpha="auto", decay_half_life=60)
        
        # 2) ARIMA sum forecast -> range constraint
        mu, low, high = self._arima_sum_forecast(horizon=1)
        sum_low = max(60, int(low) - 5)
        sum_high = min(180, int(high) + 5)
        print(f"ARIMA 预测和值区间: 目标≈{mu:.1f}, 允许范围 [{sum_low}, {sum_high}]")

        # 3) Monte Carlo candidates with entropy penalty
        raw_candidates = self._monte_carlo_candidates(p_red, p_blue, n=2500, sum_mu=mu, sum_low=sum_low, sum_high=sum_high)
        if not raw_candidates:
            print("候选为空，回退到无和值约束的采样。")
            raw_candidates = self._monte_carlo_candidates(p_red, p_blue, n=2500)

        # 4) take top-K unique
        recommendations = []
        used_red_sets = set()
        for reds, blue, score, H in raw_candidates:
            key = tuple(reds)
            if key in used_red_sets:
                continue
            odd_count = sum(1 for x in reds if x % 2)
            even_count = 6 - odd_count
            span = max(reds) - min(reds)
            total_sum = sum(reds)
            conf = float(np.mean([p_red[r-1] for r in reds]) * p_blue[blue-1])
            recommendations.append({
                'red_balls': reds,
                'blue_ball': blue,
                'description': 'LSTM+ARIMA+MonteCarlo 低熵组合',
                'strategy': 'ML低熵',
                'odd_even': f"{odd_count}奇{even_count}偶",
                'sum': total_sum,
                'span': span,
                'entropy_bits': round(H, 4),
                'confidence': round(conf, 6),
            })
            used_red_sets.add(key)
            if len(recommendations) >= num_sets:
                break

        print("\n机器学习增强推荐：")
        for i, rec in enumerate(recommendations, 1):
            red_str = " ".join([f"{x:2d}" for x in rec['red_balls']])
            print(f"推荐 {i}: {red_str} + {rec['blue_ball']:2d} | 熵:{rec['entropy_bits']:.3f}bits | 置信度:{rec['confidence']:.6f} | 和值:{rec['sum']} | 跨度:{rec['span']} | {rec['odd_even']}")

        return recommendations
    
    def _select_with_odd_even_balance(self, pool, count, existing_reds):
        """在选择时考虑奇偶平衡"""
        if count <= 0:
            return []
            
        existing_odd = sum(1 for x in existing_reds if x % 2 == 1)
        existing_even = len(existing_reds) - existing_odd
        
        # 目标：6个球中3-4个奇数比较平衡
        target_total_odd = 3 if len(existing_reds) + count <= 6 else 4
        needed_odd = max(0, target_total_odd - existing_odd)
        needed_even = count - needed_odd
        
        odd_pool = [x for x in pool if x % 2 == 1]
        even_pool = [x for x in pool if x % 2 == 0]
        
        selected = []
        
        # 选择奇数
        if needed_odd > 0 and odd_pool:
            actual_odd = min(needed_odd, len(odd_pool))
            selected.extend(random.sample(odd_pool, actual_odd))
        
        # 选择偶数
        if needed_even > 0 and even_pool:
            actual_even = min(needed_even, len(even_pool))
            selected.extend(random.sample(even_pool, actual_even))
        
        # 如果还不够，从剩余的球中补充
        while len(selected) < count and len(selected) < len(pool):
            remaining = [x for x in pool if x not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        return selected[:count]
    
    def visualize_frequency(self, save_plots=True):
        """可视化频率分析"""
        if not self.lottery_data:
            print("无数据，无法生成图表")
            return
        
        # 统计频率
        red_counter = Counter()
        blue_counter = Counter()
        
        for record in self.lottery_data:
            for red in record['red_balls']:
                red_counter[red] += 1
            blue_counter[record['blue_ball']] += 1
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 红球频率图
        red_nums = list(range(1, 34))
        red_freqs = [red_counter.get(num, 0) for num in red_nums]
        
        bars1 = ax1.bar(red_nums, red_freqs, color='red', alpha=0.7)
        ax1.set_title('红球出现频率分布', fontsize=16, fontweight='bold')
        ax1.set_xlabel('红球号码', fontsize=12)
        ax1.set_ylabel('出现次数', fontsize=12)
        ax1.set_xticks(red_nums)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, freq in zip(bars1, red_freqs):
            if freq > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(freq), ha='center', va='bottom', fontsize=8)
        
        # 蓝球频率图
        blue_nums = list(range(1, 17))
        blue_freqs = [blue_counter.get(num, 0) for num in blue_nums]
        
        bars2 = ax2.bar(blue_nums, blue_freqs, color='blue', alpha=0.7)
        ax2.set_title('蓝球出现频率分布', fontsize=16, fontweight='bold')
        ax2.set_xlabel('蓝球号码', fontsize=12)
        ax2.set_ylabel('出现次数', fontsize=12)
        ax2.set_xticks(blue_nums)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, freq in zip(bars2, blue_freqs):
            if freq > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        str(freq), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_plots:
            # 确保目录存在
            os.makedirs('pics', exist_ok=True)
            plt.savefig('pics/lottery_frequency_analysis.png', dpi=300, bbox_inches='tight')
            print("频率分析图表已保存为 pics/lottery_frequency_analysis.png")
    
    def get_lottery_rules(self):
        """获取双色球游戏规则"""
        rules = """
        === 双色球游戏规则 ===
        
        1. 号码范围：
           - 红球：01-33，选择6个号码
           - 蓝球：01-16，选择1个号码
        
        2. 中奖等级：
           一等奖：6个红球 + 1个蓝球（浮动奖金，500万元起）
           二等奖：6个红球（浮动奖金）
           三等奖：5个红球 + 1个蓝球（固定3000元）
           四等奖：5个红球 或 4个红球 + 1个蓝球（固定200元）
           五等奖：4个红球 或 3个红球 + 1个蓝球（固定10元）
           六等奖：2个红球 + 1个蓝球 或 1个红球 + 1个蓝球 或 1个蓝球（固定5元）
        
        3. 开奖时间：每周二、四、日晚21:15
        
        4. 投注方式：
           - 单式投注：手动选择号码
           - 复式投注：选择7个以上红球进行组合
           - 机选投注：系统随机选择号码
        
        5. 中奖概率：
           一等奖：1/17,721,088
           二等奖：1/1,107,568
           三等奖：1/72,107
           
        注意：彩票投注有风险，请理性购彩，量力而行！
        """
        print(rules)
    
    def generate_analysis_report(self, filename="reports/analysis_report.md"):
        """生成完整的分析报告文件"""
        print(f"正在生成分析报告: {filename}")
        
        if not self.lottery_data:
            print("无数据，无法生成报告")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 执行所有分析
        red_counter, blue_counter = self._get_frequency_analysis()
        patterns_data = self._get_patterns_analysis()
        trends_data = self._get_trends_analysis()
        recommendations = self.generate_recommendations(num_sets=8)
        
        # 生成报告内容 UTC+8时区
        current_time = (datetime.now() + timedelta(hours=8)).strftime('%Y年%m月%d日 %H:%M:%S')
        
        report_content = f"""# 🎯 双色球数据分析报告

## 📊 报告信息
- **生成时间**: {current_time} (UTC+8)
- **数据期数**: 共 {len(self.lottery_data)} 期
- **最新期号**: {self.lottery_data[0]['period'] if self.lottery_data else 'N/A'}
- **数据来源**: 中国福利彩票官方API

## ⚠️ 重要免责声明
**本分析报告仅供学习和研究使用，彩票开奖完全随机，历史数据无法预测未来结果。请理性购彩，量力而行！**

---

## 📈 最新开奖信息

"""
        
        # 添加最近5期开奖信息
        if len(self.lottery_data) >= 5:
            report_content += "### 最近5期开奖号码\n\n"
            for i, record in enumerate(self.lottery_data[:5]):
                red_str = " ".join([f"{x:02d}" for x in record['red_balls']])
                report_content += f"**{record['period']}期** ({record['date']}): {red_str} + **{record['blue_ball']:02d}**\n\n"
        
        # 添加号码频率分析
        report_content += """---

## 🔥 号码频率分析

### 红球出现频率排行榜（前15名）

| 排名 | 号码 | 出现次数 | 出现频率 |
|------|------|----------|----------|
"""
        
        red_freq = sorted(red_counter.items(), key=lambda x: x[1], reverse=True)
        for i, (num, count) in enumerate(red_freq[:15], 1):
            percentage = (count / len(self.lottery_data)) * 100
            report_content += f"| {i:02d} | **{num:02d}** | {count} | {percentage:.1f}% |\n"
        
        report_content += """
### 蓝球出现频率排行榜（前10名）

| 排名 | 号码 | 出现次数 | 出现频率 |
|------|------|----------|----------|
"""
        
        blue_freq = sorted(blue_counter.items(), key=lambda x: x[1], reverse=True)
        for i, (num, count) in enumerate(blue_freq[:10], 1):
            percentage = (count / len(self.lottery_data)) * 100
            report_content += f"| {i:02d} | **{num:02d}** | {count} | {percentage:.1f}% |\n"
        
        # 添加规律分析
        report_content += f"""
---

## 📊 号码规律分析

### 奇偶分布统计

{patterns_data['odd_even']}

### 和值分布统计

{patterns_data['sum_dist']}

### 跨度分布统计

{patterns_data['span_dist']}

---

## 📉 走势分析

### 最近10期开奖记录

{trends_data['recent_draws']}

### 热号分析

**最近10期红球热号（出现2次及以上）**: {trends_data['hot_reds']}

**最近10期蓝球热号（出现2次及以上）**: {trends_data['hot_blues']}

---

## 🎯 智能推荐号码

**⚠️ 以下推荐号码仅基于历史统计分析，不保证中奖，请理性参考！**

"""
        
        for i, rec in enumerate(recommendations, 1):
            red_str = " ".join([f"{x:02d}" for x in rec['red_balls']])
            report_content += f"**推荐组合 {i}** ({rec.get('strategy','ML低熵')}): {red_str} + **{rec['blue_ball']:02d}**\n"
            report_content += f"- 特征: {rec['odd_even']} | 和值:{rec['sum']} | 跨度:{rec['span']}\n"
            if 'entropy_bits' in rec:
                report_content += f"- 信息熵: {rec['entropy_bits']} bits\n"
            if 'confidence' in rec:
                report_content += f"- 置信度(相对): {rec['confidence']}\n"
            report_content += f"- 说明: {rec.get('description','LSTM+ARIMA+MonteCarlo 低熵组合')}\n\n"
        
        # 添加使用说明和提醒
        report_content += f"""---

## 📋 使用说明

### 数据更新频率
- 本报告每天自动更新一次
- 数据来源于中国福利彩票官方API
- 更新时间：每天晚上23:00

### 分析方法说明
1. **频率分析**: 统计每个号码在历史开奖中的出现次数
2. **规律分析**: 分析奇偶分布、和值分布、跨度分布等规律
3. **走势分析**: 观察最近期数的号码走势和热号变化
4. **智能推荐**: 基于统计概率和随机性的权重算法生成推荐号码

### 重要提醒

> 🎲 **彩票本质**: 彩票开奖具有完全的随机性和偶然性
> 
> 📊 **数据局限**: 历史数据无法预测未来开奖结果
> 
> 🎯 **参考价值**: 本分析仅供统计学习和娱乐参考
> 
> 💰 **理性购彩**: 请根据个人经济能力适度购买
> 
> ⚖️ **法律提醒**: 未满18周岁禁止购买彩票
> 
> 🏠 **家庭和睦**: 切勿因购彩影响家庭生活

---

## 📞 帮助信息

如果您或身边的人出现以下情况，请及时寻求帮助：
- 无法控制购彩行为
- 为了购彩借钱或变卖财产
- 因购彩影响工作、学习或家庭关系
- 出现焦虑、抑郁等心理问题

**全国戒赌帮助热线**: 400-161-9995

---

*报告生成时间: {current_time} (UTC+8)*  
*数据来源: 中国福利彩票官方网站*  
*仅供学习研究使用，请理性购彩*
"""
        
        # 保存报告文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            print(f"分析报告已保存到 {filename}")
        except Exception as e:
            print(f"保存分析报告失败: {e}")
    
    def _get_frequency_analysis(self):
        """内部方法：获取频率分析数据"""
        red_counter = Counter()
        blue_counter = Counter()
        
        for record in self.lottery_data:
            for red in record['red_balls']:
                red_counter[red] += 1
            blue_counter[record['blue_ball']] += 1
        
        return red_counter, blue_counter
    
    def _get_patterns_analysis(self):
        """内部方法：获取规律分析数据"""
        odd_even_dist = defaultdict(int)
        sum_dist = defaultdict(int)
        span_dist = defaultdict(int)
        
        for record in self.lottery_data:
            red_balls = record['red_balls']
            
            # 奇偶分析
            odd_count = sum(1 for x in red_balls if x % 2 == 1)
            even_count = 6 - odd_count
            odd_even_dist[f"{odd_count}奇{even_count}偶"] += 1
            
            # 和值分析
            total_sum = sum(red_balls)
            sum_range = f"{(total_sum//10)*10}-{(total_sum//10)*10+9}"
            sum_dist[sum_range] += 1
            
            # 跨度分析
            span = max(red_balls) - min(red_balls)
            span_range = f"{(span//5)*5}-{(span//5)*5+4}"
            span_dist[span_range] += 1
        
        # 格式化数据
        odd_even_result = "| 分布类型 | 出现次数 | 出现频率 |\n|----------|----------|----------|\n"
        for pattern, count in sorted(odd_even_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.lottery_data)) * 100
            odd_even_result += f"| {pattern} | {count} | {percentage:.1f}% |\n"
        
        sum_result = "| 和值范围 | 出现次数 | 出现频率 |\n|----------|----------|----------|\n"
        for sum_range, count in sorted(sum_dist.items(), key=lambda x: int(x[0].split('-')[0])):
            percentage = (count / len(self.lottery_data)) * 100
            sum_result += f"| {sum_range} | {count} | {percentage:.1f}% |\n"
        
        span_result = "| 跨度范围 | 出现次数 | 出现频率 |\n|----------|----------|----------|\n"
        for span_range, count in sorted(span_dist.items(), key=lambda x: int(x[0].split('-')[0])):
            percentage = (count / len(self.lottery_data)) * 100
            span_result += f"| {span_range} | {count} | {percentage:.1f}% |\n"
        
        return {
            'odd_even': odd_even_result,
            'sum_dist': sum_result,
            'span_dist': span_result
        }
    
    def _get_trends_analysis(self):
        """内部方法：获取趋势分析数据"""
        if len(self.lottery_data) < 10:
            return {
                'recent_draws': '数据不足',
                'hot_reds': '无',
                'hot_blues': '无'
            }
        
        recent_10 = self.lottery_data[:10]
        
        # 格式化最近10期
        recent_draws = "| 期号 | 开奖日期 | 红球号码 | 蓝球 |\n|------|----------|----------|------|\n"
        for record in recent_10:
            red_str = " ".join([f"{x:02d}" for x in record['red_balls']])
            recent_draws += f"| {record['period']} | {record['date']} | {red_str} | **{record['blue_ball']:02d}** |\n"
        
        # 冷热号分析
        red_counter = Counter()
        blue_counter = Counter()
        
        for record in recent_10:
            for red in record['red_balls']:
                red_counter[red] += 1
            blue_counter[record['blue_ball']] += 1
        
        hot_reds = [num for num, count in red_counter.items() if count >= 2]
        hot_blues = [num for num, count in blue_counter.items() if count >= 2]
        
        hot_reds_str = " ".join([f"{x:02d}" for x in sorted(hot_reds)]) if hot_reds else "无"
        hot_blues_str = " ".join([f"{x:02d}" for x in sorted(hot_blues)]) if hot_blues else "无"
        
        return {
            'recent_draws': recent_draws,
            'hot_reds': hot_reds_str,
            'hot_blues': hot_blues_str
        }
    
    def generate_aggregated_data_hjson(self, filename="data/lottery_aggregated_data.hjson"):
        """生成聚合分析数据的HJSON文件，包含详细注释供AI理解数据用途"""
        print(f"正在生成聚合数据文件: {filename}")
        
        if not self.lottery_data:
            print("无数据，无法生成聚合数据文件")
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 获取所有分析数据
        red_counter, blue_counter = self._get_frequency_analysis()
        patterns_data = self._get_patterns_analysis_raw()
        trends_data = self._get_trends_analysis_raw()
        recommendations = self.generate_recommendations(num_sets=8)
        
        # 生成时间 UTC+8
        current_time = (datetime.now() + timedelta(hours=8)).strftime('%Y年%m月%d日 %H:%M:%S')
        
        # 构建聚合数据结构
        aggregated_data = {
            "// 数据文件说明": "双色球彩票数据聚合分析结果，包含频率、规律、走势等统计数据",
            "// 文件用途": "供AI系统理解数据含义并生成相应的数据可视化图表",
            "// 更新频率": "每天自动更新一次，与开奖数据同步",
            
            "metadata": {
                "// 元数据说明": "包含数据的基本信息和统计概况",
                "lottery_type": "双色球",
                "lottery_type_en": "double_color_ball", 
                "game_rules": "红球1-33选6个，蓝球1-16选1个",
                "generated_time": current_time,
                "timezone": "UTC+8",
                "total_periods": len(self.lottery_data),
                "latest_period": self.lottery_data[0]['period'] if self.lottery_data else None,
                "latest_date": self.lottery_data[0]['date'] if self.lottery_data else None,
                "data_source": "中国福利彩票官方API"
            },
            
            "frequency_analysis": {
                "// 频率分析说明": "统计每个号码在历史开奖中的出现次数和频率",
                "// 图表建议": "适合绘制柱状图、热力图、频率分布图",
                "// 可视化用途": "展示号码冷热程度，识别高频低频号码",
                
                "red_balls": {
                    "// 红球频率数据": "红球1-33的历史出现统计",
                    "// 数据结构": "number: 号码, count: 出现次数, frequency: 出现频率(%)",
                    "data": [
                        {
                            "number": num,
                            "count": red_counter.get(num, 0),
                            "frequency": round((red_counter.get(num, 0) / len(self.lottery_data)) * 100, 2)
                        } for num in range(1, 34)
                    ],
                    "// 统计摘要": "频率分析的关键指标",
                    "summary": {
                        "highest_freq_number": max(red_counter.items(), key=lambda x: x[1])[0] if red_counter else None,
                        "highest_freq_count": max(red_counter.items(), key=lambda x: x[1])[1] if red_counter else 0,
                        "lowest_freq_number": min(red_counter.items(), key=lambda x: x[1])[0] if red_counter else None,
                        "lowest_freq_count": min(red_counter.items(), key=lambda x: x[1])[1] if red_counter else 0,
                        "average_frequency": round(sum(red_counter.values()) / len(red_counter) if red_counter else 0, 2)
                    }
                },
                
                "blue_balls": {
                    "// 蓝球频率数据": "蓝球1-16的历史出现统计", 
                    "// 数据结构": "number: 号码, count: 出现次数, frequency: 出现频率(%)",
                    "data": [
                        {
                            "number": num,
                            "count": blue_counter.get(num, 0),
                            "frequency": round((blue_counter.get(num, 0) / len(self.lottery_data)) * 100, 2)
                        } for num in range(1, 17)
                    ],
                    "// 统计摘要": "蓝球频率分析的关键指标",
                    "summary": {
                        "highest_freq_number": max(blue_counter.items(), key=lambda x: x[1])[0] if blue_counter else None,
                        "highest_freq_count": max(blue_counter.items(), key=lambda x: x[1])[1] if blue_counter else 0,
                        "lowest_freq_number": min(blue_counter.items(), key=lambda x: x[1])[0] if blue_counter else None,
                        "lowest_freq_count": min(blue_counter.items(), key=lambda x: x[1])[1] if blue_counter else 0,
                        "average_frequency": round(sum(blue_counter.values()) / len(blue_counter) if blue_counter else 0, 2)
                    }
                }
            },
            
            "pattern_analysis": {
                "// 规律分析说明": "分析号码的奇偶分布、和值分布、跨度分布等规律",
                "// 图表建议": "适合绘制饼图、堆叠柱状图、分布直方图",
                "// 可视化用途": "展示号码组合的规律性和分布特征",
                
                "odd_even_distribution": {
                    "// 奇偶分布": "红球6个号码中奇数偶数的分布情况",
                    "// 图表类型": "饼图或柱状图展示各种奇偶组合的出现频率",
                    "data": patterns_data['odd_even_dist'],
                    "total_periods": len(self.lottery_data)
                },
                
                "sum_distribution": {
                    "// 和值分布": "红球6个号码总和的分布区间统计",
                    "// 图表类型": "直方图或折线图展示和值的分布规律",
                    "// 分析意义": "帮助识别号码组合的和值趋势",
                    "data": patterns_data['sum_dist'],
                    "total_periods": len(self.lottery_data)
                },
                
                "span_distribution": {
                    "// 跨度分布": "红球最大号码与最小号码差值的分布统计",
                    "// 图表类型": "柱状图展示不同跨度范围的出现频率",
                    "// 分析意义": "反映号码选择的分散程度",
                    "data": patterns_data['span_dist'],
                    "total_periods": len(self.lottery_data)
                }
            },
            
            "trend_analysis": {
                "// 走势分析说明": "分析最近期数的号码走势和热号变化",
                "// 图表建议": "适合绘制时间序列图、热力图、趋势线图",
                "// 可视化用途": "展示短期内号码的冷热变化趋势",
                "// 分析周期": "最近10期开奖数据",
                
                "recent_draws": trends_data['recent_draws'],
                "hot_numbers": {
                    "// 热号定义": "最近10期中出现2次及以上的号码",
                    "// 图表类型": "标记图或高亮显示热号在走势图中的位置",
                    "red_hot_numbers": trends_data['hot_reds'],
                    "blue_hot_numbers": trends_data['hot_blues']
                }
            },
            
            "recommendations": {
                "// 推荐号码说明": "基于历史统计分析生成的8种策略推荐组合",
                "// 图表建议": "表格展示或卡片式布局展示推荐组合",
                "// 重要提醒": "仅供参考，彩票开奖完全随机",
                "// 策略说明": "包含高频主导、均衡分布、冷热结合等多种选号策略",
                
                "strategies": [
                    {
                        "strategy_name": rec['strategy'],
                        "description": rec['description'],
                        "red_balls": rec['red_balls'],
                        "blue_ball": rec['blue_ball'],
                        "characteristics": {
                            "odd_even_ratio": rec['odd_even'],
                            "sum_value": rec['sum'],
                            "span_value": rec['span']
                        }
                    } for rec in recommendations
                ],
                
                "strategy_summary": {
                    "total_strategies": len(recommendations),
                    "strategy_types": [rec['strategy'] for rec in recommendations]
                }
            },
            
            "visualization_suggestions": {
                "// 可视化建议": "针对不同数据类型的图表绘制建议",
                
                "frequency_charts": {
                    "chart_types": ["bar_chart", "heatmap", "bubble_chart"],
                    "description": "频率数据适合用柱状图展示排名，热力图展示分布，气泡图展示频率大小"
                },
                
                "pattern_charts": {
                    "chart_types": ["pie_chart", "stacked_bar", "histogram"],
                    "description": "规律数据适合用饼图展示比例，堆叠柱状图展示分类，直方图展示分布"
                },
                
                "trend_charts": {
                    "chart_types": ["line_chart", "scatter_plot", "timeline"],
                    "description": "走势数据适合用折线图展示变化，散点图展示分布，时间轴展示历史"
                },
                
                "recommendation_display": {
                    "display_types": ["table", "card_layout", "grid_view"],
                    "description": "推荐数据适合用表格展示详情，卡片布局展示策略，网格视图展示组合"
                }
            }
        }
        
        # 保存HJSON文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                hjson.dump(aggregated_data, f, ensure_ascii=False, indent=2)
            print(f"聚合数据文件已保存到 {filename}")
        except Exception as e:
            print(f"保存聚合数据文件失败: {e}")
    
    def _get_patterns_analysis_raw(self):
        """内部方法：获取原始规律分析数据"""
        odd_even_dist = defaultdict(int)
        sum_dist = defaultdict(int)
        span_dist = defaultdict(int)
        
        for record in self.lottery_data:
            red_balls = record['red_balls']
            
            # 奇偶分析
            odd_count = sum(1 for x in red_balls if x % 2 == 1)
            even_count = 6 - odd_count
            odd_even_dist[f"{odd_count}奇{even_count}偶"] += 1
            
            # 和值分析
            total_sum = sum(red_balls)
            sum_range = f"{(total_sum//10)*10}-{(total_sum//10)*10+9}"
            sum_dist[sum_range] += 1
            
            # 跨度分析
            span = max(red_balls) - min(red_balls)
            span_range = f"{(span//5)*5}-{(span//5)*5+4}"
            span_dist[span_range] += 1
        
        return {
            'odd_even_dist': dict(odd_even_dist),
            'sum_dist': dict(sum_dist),
            'span_dist': dict(span_dist)
        }
    
    def _get_trends_analysis_raw(self):
        """内部方法：获取原始趋势分析数据"""
        if len(self.lottery_data) < 10:
            return {
                'recent_draws': [],
                'hot_reds': [],
                'hot_blues': []
            }
        
        recent_10 = self.lottery_data[:10]
        
        # 最近10期数据
        recent_draws = []
        for record in recent_10:
            recent_draws.append({
                'period': record['period'],
                'date': record['date'],
                'red_balls': record['red_balls'],
                'blue_ball': record['blue_ball']
            })
        
        # 冷热号分析
        red_counter = Counter()
        blue_counter = Counter()
        
        for record in recent_10:
            for red in record['red_balls']:
                red_counter[red] += 1
            blue_counter[record['blue_ball']] += 1
        
        hot_reds = [num for num, count in red_counter.items() if count >= 2]
        hot_blues = [num for num, count in blue_counter.items() if count >= 2]
        
        return {
            'recent_draws': recent_draws,
            'hot_reds': sorted(hot_reds),
            'hot_blues': sorted(hot_blues)
        }
    
    def update_readme_recommendations(self, readme_path="README.md", timestamp=None):
        """更新README.md中的推荐号码"""
        print(f"正在更新README.md中的双色球推荐号码...")
        
        if not self.lottery_data:
            print("无数据，无法更新README推荐号码")
            return
        
        try:
            # 生成推荐号码
            recommendations = self.generate_recommendations(num_sets=5)

            # 读取现有README内容或创建最小骨架
            if not os.path.exists(readme_path):
                print(f"ℹ️  README 不存在：将创建最小 README 并插入推荐区。")
                content = "# 🎯 双色球开奖数据分析系统\n\n> 本仓库为历史数据分析与可视化，仅供学习研究使用。\n\n"
            else:
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()

            # 使用传入的时间戳或生成新的时间戳 UTC+8
            if timestamp:
                current_time = timestamp
            else:
                current_time = (datetime.now() + timedelta(hours=8)).strftime('%Y年%m月%d日 %H:%M:%S')

            # 构建推荐号码内容（保留你原来的模板，但不会因文件缺失失败）
            recommendations_content = f"""## 🎯 今日推荐号码

        **⚠️ 以下推荐号码基于历史统计分析，仅供参考，不保证中奖！**

        ### 双色球推荐 (更新时间: {current_time})

        """
            for i, rec in enumerate(recommendations, 1):
                red_str = " ".join([f"{x:02d}" for x in rec['red_balls']])
                recommendations_content += f"**推荐 {i}** ({rec['strategy']}): `{red_str}` + `{rec['blue_ball']:02d}`  \n"
                recommendations_content += f"*{rec['description']} | {rec['odd_even']} | 和值:{rec['sum']} | 跨度:{rec['span']}*\n\n"

            # 查找第二个H2标题的位置（免责声明后）
            lines = content.split('\n')
            h2_count = 0
            insert_index = -1

            for i, line in enumerate(lines):
                if line.startswith('## '):
                    h2_count += 1
                    if h2_count == 2:  # 第二个H2标题
                        insert_index = i
                        break

            if insert_index == -1:
                print("未找到合适的插入位置，将在文件末尾添加")
                new_content = content.rstrip() + "\n\n" + recommendations_content
            else:
                # 检查是否已存在推荐号码部分
                existing_rec_index = -1
                for i in range(insert_index, len(lines)):
                    if "今日推荐号码" in lines[i]:
                        existing_rec_index = i
                        break

                if existing_rec_index != -1:
                    # 找到推荐号码部分的结束位置
                    end_index = existing_rec_index
                    for i in range(existing_rec_index + 1, len(lines)):
                        if lines[i].startswith('## '):
                            end_index = i
                            break
                    else:
                        end_index = len(lines)

                    new_lines = lines[:existing_rec_index] + recommendations_content.strip().split('\n') + lines[end_index:]
                else:
                    new_lines = lines[:insert_index] + recommendations_content.strip().split('\n') + [''] + lines[insert_index:]

                new_content = '\n'.join(new_lines)

            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"README.md中的双色球推荐号码已更新")

        except Exception as e:
            print(f"更新README推荐号码失败: {e}")

def main():
    """主函数"""
    # 显示免责声明
    print("=" * 80)
    print(f"🧰 Runtime -> Python: {sys.version.split()[0]} | Torch: {getattr(torch, '__version__', 'N/A')} | CUDA: {torch.cuda.is_available() if hasattr(torch, 'cuda') else False}")
    print(f"📄 Running script: {__file__}")
    if sys.version_info < (3, 11):
        print("⚠️ 建议使用 Python 3.11+ 以避免 macOS LibreSSL/urllib3 警告，并获得更好的依赖兼容性。")
    print("🎯 双色球数据分析系统")
    print("=" * 80)
    print("⚠️  重要免责声明：")
    print("• 彩票开奖完全随机，历史数据无法预测未来")
    print("• 本分析仅供学习参考，不构成投注建议")
    print("• 请理性购彩，量力而行，未满18周岁禁止购买")
    print("• 使用本软件产生的任何后果由用户自行承担")
    print("=" * 80)
    
    analyzer = DoubleColorBallAnalyzer()
    
    print("\n双色球开奖数据分析系统")
    print("=" * 50)
    
    # 始终抓取最新数据，覆盖现有文件
    print("⚠️  正在抓取最新数据，请确保网络连接正常...")
    max_pages = analyzer.get_max_pages()
    analyzer.fetch_lottery_data(max_pages=max_pages)
    analyzer.save_data()
    
    if not analyzer.lottery_data:
        print("❌ 无法获取数据，程序退出")
        return
    
    # 显示游戏规则
    analyzer.get_lottery_rules()
    
    # 执行各种分析
    red_counter, blue_counter = analyzer.analyze_frequency()
    analyzer.analyze_patterns()
    analyzer.analyze_trends()
    # 训练机器学习模型（LSTM）
    try:
        analyzer.train_ml_model(seq_len=10, epochs=5, lr=1e-3, hidden_size=64, dropout=0.2)
    except Exception as e:
        print(f"⚠️  模型训练失败: {e}，将使用时间衰减频率作为备选。")
        analyzer.trained = False
    print(f"🧪 模型训练状态: {'已训练(使用LSTM融合)' if analyzer.trained else '未训练(使用时间衰减频率)'}")
    analyzer.generate_recommendations(num_sets=8)
    # 生成推荐号码
    # recommendations = analyzer.generate_recommendations(num_sets=5)
    
    # 生成可视化图表
    try:
        analyzer.visualize_frequency()
    except Exception as e:
        print(f"⚠️  图表生成失败: {e}")
        print("可能是字体问题，请检查系统中文字体支持")
    
    # 生成分析报告
    analyzer.generate_analysis_report()
    
    # 生成聚合数据文件
    analyzer.generate_aggregated_data_hjson()
    
    # 更新README.md中的推荐号码
    analyzer.update_readme_recommendations()
    
    print("\n" + "=" * 50)
    print("📋 重要提醒：")
    print("• 以上推荐号码基于历史统计，仅供参考")
    print("• 彩票具有偶然性，请勿过度依赖任何预测")
    print("• 理性购彩，适度娱乐，珍惜家庭和睦")
    print("• 如有赌博问题，请寻求专业帮助")
    print("=" * 50)
    print("✅ 分析完成！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查网络连接和依赖库安装情况") 