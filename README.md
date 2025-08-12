# 🎯 Double Chromosphere Lottery Data Analysis System

# 🎯 双色球开奖数据分析系统

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚠️ Important Disclaimer

## ⚠️ 重要免责声明

**This project is for technical learning and data analysis research purposes only.**  
**本项目仅用于技术学习和数据分析研究目的。**

- 🎲 Lottery results are completely random; historical data cannot predict future outcomes.  
  彩票开奖结果完全随机，历史数据无法预测未来结果。
- 📊 The analysis is for reference only and does not constitute any betting advice.  
  本分析结果仅供参考，不构成任何投注建议。
- 💰 Please gamble responsibly and within your means. Minors under 18 are prohibited from purchasing lottery tickets.  
  请理性购彩，量力而行，未满 18 周岁禁止购买彩票。
- ⚖️ The developer is not responsible for any losses resulting from using this script.  
  开发者不承担因使用本脚本产生的任何损失。
- 🤖 **The addition of machine learning models does not guarantee prediction accuracy.**  
  机器学习模型的引入并不保证预测准确率。


---

## 🚀 Features

## 🚀 功能特性

- 📈 **Automatic Data Fetching**: Daily fetch of the latest lottery draw data  
  **自动数据抓取**：每日自动抓取最新双色球开奖数据
- 📊 **Statistical Analysis**: Multi-dimensional analysis of number frequency, odd/even distribution, sum, and span  
  **统计分析**：号码频率、奇偶分布、和值跨度等多维度分析
- 📉 **Trend Analysis**: Hot/cold number analysis and trend detection  
  **趋势分析**：冷热号码分析和走势识别
- 🎯 **Intelligent Recommendation**: Number recommendation algorithm based on statistics  
  **智能推荐**：基于统计学的号码推荐算法
- 📱 **Visualization**: Generate intuitive frequency analysis charts  
  **可视化图表**：生成直观的频率分析图表
- 📋 **Analysis Report**: Automatically generate detailed Markdown analysis reports  
  **分析报告**：自动生成详细的 Markdown 格式分析报告
- 🤖 **Automated Deployment**: GitHub Actions for automatic runs and data updates  
  **自动化部署**：GitHub Actions 自动运行和数据更新
- 🧠 **LSTM-based Machine Learning Model**: Sequence-to-sequence prediction blending historical frequency analysis with LSTM deep learning  
  **基于 LSTM 的机器学习模型**：结合历史频率分析与 LSTM 深度学习的序列预测
- 🟢 **Entropy Minimization**: Optimize number selection by minimizing entropy in historical draws  
  **熵最小化**：通过最小化历史开奖熵优化选号
- 🎲 **Monte Carlo Simulation**: Simulate large numbers of draws to evaluate strategies  
  **蒙特卡洛模拟**：大量开奖模拟以评估策略
- 📈 **ARIMA Trend Forecasting**: Use ARIMA models for time series trend prediction  
  **ARIMA 趋势预测**：利用 ARIMA 模型进行时间序列趋势预测
- 🧪 **Rolling Backtesting**: Evaluate strategies and ML models with rolling backtest, Hit@k, and blue-ball hit rate metrics  
  **滚动回测**：用回测、Hit@k 和蓝球命中率等指标评估策略与模型
- 🔒 **Strict ML Training Mode**: Enable strict mode for ML model training during backtesting  
  **严格模式**：回测时可启用机器学习模型的严格训练模式

---

## 📁 Project Structure

## 📁 项目结构

```
lucky_ball/
├── lottery_analyzer.py          # Main analysis script | 主分析脚本
├── requirements.txt             # Python dependencies | Python依赖包
├── lottery_data.json            # Draw data file (auto-generated) | 开奖数据文件 (自动生成)
├── analysis_report.md           # Detailed analysis report (auto-generated) | 详细分析报告 (自动生成)
├── lottery_frequency_analysis.png # Analysis chart (auto-generated) | 分析图表 (自动生成)
├── backtest.py                  # Backtesting and ML evaluation script | 回测与机器学习评估脚本
├── .github/workflows/
│   └── update-lottery-data.yml  # GitHub Actions workflow | GitHub Actions工作流
├── README.md                    # Project documentation | 项目说明
├── LICENSE                      # Open source license | 开源协议
└── .gitignore                   # Git ignore file | Git忽略文件
```

---

## 🛠️ Installation and Usage

## 🛠️ 安装使用

### Local Run

### 本地运行

1. **Clone the repository**  
   **克隆仓库**
   ```bash
   git clone https://github.com/your-username/lucky_ball.git
   cd lucky_ball
   ```
2. **Install dependencies**  
   **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run analysis**  
   **运行分析**
   ```bash
   python lottery_analyzer.py
   ```

### Backtesting and ML Evaluation

### 回测与机器学习评估

Run rolling backtest and compare ML models with baseline:  
运行滚动回测，比较机器学习模型与基线方法：

```bash
python backtest.py --model lstm --window 50 --recommend 5 --strict
```

- `--model lstm` Use LSTM-based ML model  
  `--model lstm` 使用 LSTM 机器学习模型
- `--window 50` Rolling window size  
  `--window 50` 滚动窗口大小
- `--recommend 5` Number of recommendations per period  
  `--recommend 5` 每期推荐组数
- `--strict` Enable strict ML training mode (train only on past data in each window)  
  `--strict` 启用严格模式（每个窗口仅用历史数据训练）

Results include Hit@k and blue-ball hit rate for both ML and Baseline strategies.  
结果将输出 ML 与基线策略的 Hit@k 和蓝球命中率等指标。

### GitHub Actions Automation

### GitHub Actions 自动化

This project is configured to run automatically via GitHub Actions:  
本项目已配置 GitHub Actions 自动运行：

- 🕐 **Scheduled Runs**: Fetch latest data daily at 23:00 (UTC+8)  
  **定时运行**：每天晚上 23:00(UTC+8)自动抓取最新数据
- 🖱️ **Manual Trigger**: Trigger runs manually on the Actions page  
  **手动触发**：可在 Actions 页面手动触发运行
- 📝 **Auto Commit**: Automatically commit new data to the repository  
  **自动提交**：有新数据时自动提交到仓库
- 🏷️ **Release Creation**: Create a release with data files on daily updates  
  **创建发布**：每日数据更新时自动创建带数据文件的 release

---

## 📊 Analysis Capabilities

## 📊 分析功能

### 1. Number Frequency Analysis

### 1. 号码频率分析

- Red and blue ball frequency statistics  
  红球和蓝球的出现频率统计
- Hot and cold number identification  
  热号和冷号识别
- Visualization of frequency distribution  
  可视化频率分布图

### 2. Pattern Analysis

### 2. 号码规律分析

- Odd/even distribution  
  奇偶分布规律
- Sum statistics  
  和值分布统计
- Span analysis  
  跨度分布分析

### 3. Trend Analysis

### 3. 走势分析

- Recent period trends  
  最近期数走势
- Hot/cold number changes  
  冷热号码变化
- Number omission statistics  
  号码遗漏统计

### 4. Intelligent Recommendation

### 4. 智能推荐

- Probability-based recommendation  
  基于概率统计的号码推荐
- Multiple set generation  
  多组号码生成
- Weight algorithm optimization  
  权重算法优化

### 5. Analysis Report

### 5. 分析报告

- Auto-generate Markdown report  
  自动生成 Markdown 格式报告
- Complete statistical analysis data  
  包含完整的统计分析数据
- Detailed instructions and risk warnings  
  提供详细的使用说明和风险提醒
- Daily automatic update  
  每日自动更新

### 6. Machine Learning & Backtest

### 6. 机器学习与回测

- LSTM-based prediction blending historical frequency  
  基于 LSTM 的历史频率混合预测
- Entropy minimization and ARIMA trend forecasting  
  熵最小化与 ARIMA 趋势预测
- Monte Carlo simulation for strategy evaluation  
  蒙特卡洛模拟评估策略
- Rolling backtest with Hit@k and blue-ball hit rate metrics  
  滚动回测，输出 Hit@k 和蓝球命中率等指标
- Strict ML training mode for robust evaluation  
  严格模式下的机器学习模型评估

---

## 🔧 Configuration

## 🔧 配置说明

### Modify Fetch Parameters

### 修改抓取参数

In `lottery_analyzer.py`, you can adjust the following parameters:  
在 `lottery_analyzer.py` 中可调整如下参数：

```python
# Change request headers
self.headers = {
    'User-Agent': '...'  # Update as needed | 可根据需要更新
}
# Change number of recommendation sets
recommendations = analyzer.generate_recommendations(num_sets=5)
```

### Modify GitHub Actions Schedule

### 修改 GitHub Actions 运行时间

Edit the cron expression in `.github/workflows/update-lottery-data.yml`:  
在 `.github/workflows/update-lottery-data.yml` 中修改 cron 表达式：

```yaml
schedule:
  # 23:00 (UTC+8)
  - cron: "0 15 * * *"
```

---

## 📈 Data Source

## 📈 数据来源

Data is sourced from the official China Welfare Lottery API:  
数据来源于中国福利彩票官方网站 API：

- **API URL**: `https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice`
- **Data format**: JSON  
  **数据格式**: JSON
- **Update frequency**: After draws on Tuesday, Thursday, and Sunday  
  **更新频率**: 每周二、四、日开奖后更新

---

## 🤝 Contribution Guide

## 🤝 贡献指南

Contributions are welcome via Issues and Pull Requests!  
欢迎提交 Issue 和 Pull Request！

1. Fork the repository  
   Fork 本仓库
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)  
   创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
   提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)  
   推送到分支 (`git push origin feature/AmazingFeature`)
5. Open a Pull Request  
   打开 Pull Request

---

## 📄 License

## 📄 开源协议

This project is licensed under the [MIT License](LICENSE).  
本项目基于 [MIT License](LICENSE) 开源协议。

---

## 🙏 Acknowledgements

## 🙏 致谢

- **Original Author**: [snjyor](https://github.com/snjyor) 
   Original version available at [https://github.com/snjyor/lucky_ball](https://github.com/snjyor/lucky_ball)  
   原作者：[snjyor](https://github.com/snjyor) 
   原始版本可在 [https://github.com/snjyor/lucky_ball](https://github.com/snjyor/lucky_ball) 获取
- Thanks to the official China Welfare Lottery for open data  
  感谢中国福利彩票官方提供的开放数据
- Thanks to all open source contributors and libraries  
  感谢所有开源贡献者的工具和库

---

## ⚖️ Legal Statement

## ⚖️ 法律声明

- This project strictly complies with relevant laws and regulations  
  本项目严格遵守相关法律法规
- For technical research and learning only  
  仅用于技术研究和学习交流
- No encouragement of any form of gambling  
  不鼓励任何形式的赌博行为
- All consequences of illegal use are at your own risk  
  如有违法违规使用，后果自负

---

**Remember: Lottery is risky, gamble with caution! Play responsibly, live happily!** 🍀  
**记住：彩票有风险，投注需谨慎！理性购彩，快乐生活！** 🍀
