# 分岔 / FENCHA

**A historical-data forecasting engine for life decisions and world events.**

分岔不是聊天式“算命”，也不是只预测足球。它把任何预测问题转成一个可回测、可校准、可追责的时间序列任务：只允许使用预测时点之前能够获得的证据，用历史相似案例形成基础概率，并永久保存每一次概率修改。

## 当前能力

- **时间防泄漏**：每条特征都带 `observed_at`，训练和预测均拒绝使用截止时点之后的信息。
- **历史类比训练**：对数值、类别、布尔特征计算相似度，用历史已结算案例形成加权概率。
- **走步回测**：按时间顺序逐案预测，绝不随机打乱过去与未来。
- **概率评分**：Brier score、log loss、分箱校准误差。
- **预测账本**：问题、证据、概率版本和结算均追加保存；更新不会覆盖旧预测。
- **首个真实数据构造器**：把 ParlGov 内阁史转成“政府首脑未来 180 天是否离任”的月度历史预测截面。
- **领域无关**：同一套结构可承载地缘政治、宏观经济、职业选择、升学、企业经营与体育问题。

## 核心原则

1. **先定义可结算问题，再谈预测。**
2. **当时不知道的信息，回测时也不能知道。**
3. **输出概率，不输出神谕。**
4. **每个概率都能追溯到基础率、相似案例和证据快照。**
5. **允许模型被现实打脸，但不允许事后改历史。**
6. **跨领域共用训练框架，领域特征与结算规则独立配置。**

## 快速运行

需要 Python 3.11+。

```bash
python -m pip install -e '.[dev]'
pytest -q
python -m fencha demo
```

`demo` 会构造一批按年份排列的历史案例，执行严格走步回测，并把一次新预测写入本地 SQLite 账本。

## 生成第一张真实训练表

ParlGov 收录 established democracies 的政党、选举和内阁历史。FENCHA 会把连续由同一首相领导的多届内阁合并为一个领导人任期，再在每个月初提出一次严格可结算的问题：

> 这位政府首脑会不会在未来 180 天内离任？

```bash
python -m fencha build-parlgov
```

默认命令会：

1. 下载官方 ParlGov CSV 压缩包；
2. 保存原始快照并计算 SHA-256；
3. 聚合同一内阁的执政党和席位；
4. 合并同一首相连续领导的多届内阁；
5. 生成月度预测截面和 180 天标签；
6. 删除观察窗口尚未完整结束的右删失样本；
7. 输出 JSONL 案例表和构建清单。

默认输出：

```text
data/raw/parlgov/
data/processed/parlgov_leader_exit_180d.jsonl
data/processed/parlgov_leader_exit_180d.manifest.json
```

可以用本地快照复现：

```bash
python -m fencha build-parlgov \
  --source data/raw/parlgov/parlgov-development_csv-utf-8.zip \
  --as-of 2023-06-30
```

当前结构特征包括：任期长度、当前内阁年龄、距上次选举天数、联合政府党数、看守政府状态、内阁类型、执政席位占比和少数政府标记。

## 训练闭环

```text
历史事件与当时可见证据
        ↓
统一案例结构 + 时间戳校验
        ↓
按预测日期排序
        ↓
只用更早案例训练
        ↓
生成概率 + 相似案例解释
        ↓
现实结算
        ↓
Brier / 校准 / 分领域误差
        ↓
更新特征、权重和问题定义
```

## 数据路线

1. **ParlGov**：政府首脑任期、选举、联盟与议会席位，已经进入代码。
2. **GDELT**：全球事件、行动者、地点、事件类型与媒体语调。
3. **World Bank / FRED / OECD**：国家和宏观时间序列。
4. **Metaculus / Polymarket 已结算问题**：概率轨迹、问题定义与结算结果。
5. **用户私有人生案例库**：只保存用户主动录入的决策、条件和结果，不抓取他人隐私。

详见 [`docs/DATA_STRATEGY.md`](docs/DATA_STRATEGY.md) 与 [`docs/MODEL_DESIGN.md`](docs/MODEL_DESIGN.md)。

## 当前边界

ParlGov 首版只覆盖其收录的 established democracies，且“内阁变化”不总等于“首相离任”，所以构造器专门合并同一领导人连续内阁。它目前提供制度和任期结构特征，尚未接入 GDELT 舆情/冲突特征与宏观数据。真实预测价值必须由冻结时间盲测证明。
