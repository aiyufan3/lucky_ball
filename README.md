# Lottery Data Analysis System | 彩票数据分析系统

---

## Project Introduction | 项目介绍

Welcome to the **Lottery Data Analysis System**, a comprehensive tool designed for analyzing and predicting lottery draws with a focus on **Double Chromosphere (双色球)** and **Happy 8 (快乐8)** games.  
欢迎使用**彩票数据分析系统**，这是一个专注于**双色球**和**快乐8**彩票的综合分析与预测工具。

This system integrates advanced statistical analysis, machine learning models (LSTM, ARIMA), Monte Carlo simulations, and entropy minimization techniques to provide intelligent number recommendations and detailed insights.  
本系统结合先进的统计分析、机器学习模型（LSTM、ARIMA）、蒙特卡洛模拟及熵最小化技术，提供智能选号推荐及详细数据洞察。

---

## Today’s Recommendations | 今日推荐号码

**Recommendations are based on historical data analysis and do not guarantee winnings. Please play responsibly!**  
**以下推荐号码基于历史数据分析，仅供参考，不保证中奖！请理性购彩！**

| Game | Recommendation # | Red Balls / Numbers | Blue Ball / Extra | Notes / Features                                  |
|-------|------------------|---------------------|-------------------|--------------------------------------------------|
| **Double Chromosphere (双色球)** | 1                | 02 06 14 16 31 33   | 06                | LSTM+ARIMA+MonteCarlo low entropy | 2 odd, 4 even | Sum:102 | Span:31 |
| **Double Chromosphere (双色球)** | 2                | 02 06 10 14 20 23   | 05                | LSTM+ARIMA+MonteCarlo low entropy | 1 odd, 5 even | Sum:75  | Span:21 |
| **Double Chromosphere (双色球)** | 3                | 06 10 14 15 28 31   | 12                | LSTM+ARIMA+MonteCarlo low entropy | 2 odd, 4 even | Sum:104 | Span:25 |
| **Double Chromosphere (双色球)** | 4                | 02 06 08 14 15 26   | 06                | LSTM+ARIMA+MonteCarlo low entropy | 1 odd, 5 even | Sum:71  | Span:24 |
| **Double Chromosphere (双色球)** | 5                | 02 06 10 13 22 31   | 11                | LSTM+ARIMA+MonteCarlo low entropy | 2 odd, 4 even | Sum:84  | Span:29 |
| **Happy 8 (快乐8)**              | 1                | 03 07 12 18 22 27 31 35 | -                 | Monte Carlo simulation with entropy minimization |
| **Happy 8 (快乐8)**              | 2                | 01 05 09 14 20 25 29 33 | -                 | Hot/cold number trend analysis                     |
| **Happy 8 (快乐8)**              | 3                | 04 08 13 17 21 26 30 34 | -                 | LSTM-based prediction with ARIMA trend forecasting |

---

## Features | 功能特性

| English Description                                    | 中文描述                                         |
|--------------------------------------------------------|------------------------------------------------|
| Automatic daily data fetching                           | 自动每日抓取最新开奖数据                         |
| Multi-dimensional statistical analysis                 | 多维度统计分析（频率、奇偶、和值、跨度等）        |
| Trend and hot/cold number analysis                      | 趋势分析及冷热号码识别                          |
| Intelligent recommendation algorithms                   | 智能号码推荐算法                                |
| Visualization charts for frequency and trends          | 频率与趋势可视化图表                            |
| Auto-generated detailed Markdown analysis reports       | 自动生成详细的 Markdown 分析报告                  |
| Machine learning models: LSTM, ARIMA, entropy minimization | 机器学习模型：LSTM、ARIMA、熵最小化              |
| Monte Carlo simulation for strategy evaluation          | 蒙特卡洛模拟策略评估                            |
| Rolling backtesting with Hit@k and blue-ball hit rate   | 滚动回测，输出 Hit@k 和蓝球命中率指标            |
| Strict ML training mode for robust evaluation           | 严格模式训练，确保模型评估的鲁棒性                |
| Support for both Double Chromosphere and Happy 8 games  | 同时支持双色球与快乐8彩票分析与推荐                |
| Scheduled automation with GitHub Actions                 | GitHub Actions 定时自动化运行                    |

---

## Installation & Usage | 安装与使用

### Local Setup | 本地安装

1. Clone the repository | 克隆仓库  
   ```bash
   git clone https://github.com/your-username/lucky_ball.git
   cd lucky_ball
   ```
2. Install dependencies | 安装依赖  
   ```bash
   pip install -r requirements.txt
   ```
3. Run analysis | 运行分析  
   ```bash
   python lottery_analyzer.py
   ```

### Backtesting & ML Evaluation | 回测与机器学习评估

Run rolling backtest to evaluate ML and baseline models:  
运行滚动回测，评估机器学习模型与基线模型：

```bash
python backtest.py --model lstm --window 50 --recommend 5 --strict
```

- `--model lstm`: Use LSTM-based machine learning model | 使用基于 LSTM 的机器学习模型  
- `--window 50`: Rolling window size | 滚动窗口大小  
- `--recommend 5`: Number of recommendation sets per period | 每期推荐组数  
- `--strict`: Enable strict training mode (train only on past data) | 启用严格训练模式（仅用历史数据训练）

Results include Hit@k and blue-ball hit rate for both ML and baseline strategies.  
结果包含机器学习和基线策略的 Hit@k 及蓝球命中率指标。

---

## Automation with GitHub Actions | GitHub Actions 自动化

This project supports automated data fetching, analysis, and report generation via GitHub Actions:  
本项目通过 GitHub Actions 实现自动数据抓取、分析及报告生成：

| Automation Type      | Description                                     | 说明                                               |
|----------------------|-------------------------------------------------|----------------------------------------------------|
| Scheduled Runs    | Daily at 23:00 (UTC+8) fetch latest data       | 每天晚上23:00（UTC+8）自动抓取最新开奖数据          |
| Manual Trigger    | Trigger runs manually via GitHub Actions page  | 可在 GitHub Actions 页面手动触发运行                |
| Auto Commit       | Automatically commit updated data files         | 自动提交更新后的数据文件                             |
| Release Creation | Create releases with data on daily updates      | 每日数据更新时自动创建包含数据文件的 Release         |

Modify scheduling by editing `.github/workflows/update-lottery-data.yml`.  
可通过编辑 `.github/workflows/update-lottery-data.yml` 修改定时任务。

---

## Data Source | 数据来源

Official China Welfare Lottery API powers the data:  
数据来源于中国福利彩票官方网站 API：

- **API URL**: `https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice`  
- **Format**: JSON  
- **Update Frequency**: After draws on Tuesday, Thursday, and Sunday  

---

## Contribution | 贡献指南

Contributions are warmly welcomed! Please follow the standard GitHub workflow:  
欢迎贡献！请遵循标准 GitHub 工作流程：

1. Fork the repository | Fork 仓库  
2. Create a feature branch (`git checkout -b feature/AmazingFeature`) | 创建特性分支  
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`) | 提交更改  
4. Push to the branch (`git push origin feature/AmazingFeature`) | 推送分支  
5. Open a Pull Request | 创建 Pull Request  

---

## License & Legal Notice | 许可与法律声明

This project is licensed under the [MIT License](LICENSE).  
本项目基于 [MIT License](LICENSE) 开源协议。

---

### Important Disclaimer | 重要免责声明

- This project is for **technical learning and data analysis research purposes only**.  
  本项目仅用于技术学习和数据分析研究目的。  
- Lottery results are completely random; historical data cannot predict future outcomes.  
  彩票开奖结果完全随机，历史数据无法预测未来结果。  
- Analysis is for reference only and does not constitute betting advice.  
  本分析结果仅供参考，不构成任何投注建议。  
- Please gamble responsibly and within your means; minors under 18 are prohibited from purchasing lottery tickets.  
  请理性购彩，量力而行，未满18周岁禁止购买彩票。  
- The developer is not responsible for any losses arising from use of this software.  
  开发者不承担因使用本软件产生的任何损失。  
- Machine learning models do not guarantee prediction accuracy.  
  机器学习模型的引入并不保证预测准确率。  
- This project complies strictly with all relevant laws and regulations and does not encourage gambling.  
  本项目严格遵守相关法律法规，不鼓励任何形式的赌博行为。  
- Any illegal use is at your own risk.  
  如有违法违规使用，后果自负。  

---

## Acknowledgements | 致谢

- Original author: [snjyor](https://github.com/snjyor)  
  原作者：[snjyor](https://github.com/snjyor)  
- Official China Welfare Lottery for open data support  
  感谢中国福利彩票官方提供的开放数据  
- All open source contributors and libraries used in this project  
  感谢所有开源贡献者及所使用的开源库  

---

**Remember: Lottery is risky, gamble with caution! Play responsibly, live happily!**  
**记住：彩票有风险，投注需谨慎！理性购彩，快乐生活！**