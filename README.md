# DSAI 2022 Final Project - Predict-Future-Sales

## Overview
### NCKU DSAI Final Project - Predict Future Sales

**Kaggle:** [Predict Future Sales](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/leaderboard)

根據其提供的每日歷史銷售數據，來為測試集預測每個商店銷售的產品總量。

由於商店和產品列表每個月都會略有變化，因此我們創建一個可以處理此類情況的模型。

## Data
### Training Data
* 2013 年 1 月到 2015 年 10 月的每日銷售資訊
* 60 間商店
* 22171 項商品
* 84 種產品類型

### Testing Data
* 2015 年 11 月的商店名稱及商品名稱 → 預測出商品銷售金額

## Data Analysis
* 兩年冬季的銷售量較高 → 產品具有**季節性**
* 從 2013 年至 2015 年，產品銷售量具有**下降**的趨勢
* 商品價格可能會影響銷售量
* 城市消費力會影響銷售量


<img src="https://github.com/hardychang/DSAI2022_Final-Predict-Future-Sales/blob/main/trend.png" width="600"/><br/>

## Feature Selection
* **每月商品總銷售量**
  * 將每日商品銷售量總和，存入新增的每月商品銷售量的欄位
* **每月商品平均銷售量**
  * 將每月銷售量取平均值存平均銷售量的欄位中
* **城市地點**
  * 從資料可以發現商店名稱中包含城市名稱
  * 選取出城市名稱並作編號後存入城市編號欄位中
* **季節性特徵**
  * 每年的 10 月到隔年 2 月銷售量較突出
* **每月商店總銷售額**
  * 商品的收益為銷售數量 * 商品價格
  * 將每日商品收益加總成每月的總銷售額，存入新增的欄位
* **每月商店平均銷售額**
  * 每月的總銷售額取平均值存平均銷售額的欄位中
* **每月周末數**
  * 周末時段可能有較多的消費機會
* **剛上市的商品與已上市的商品**
  * 剛上市的商品無 2015 以前的歷史資料，而已上市商品有過去歷史資料可提供時間序列的關聯性
* **相似商品的銷售額**
  * 資料集的數據可能是根據商品名稱排序使整體月銷售量呈現趨勢，因此嘗試計算訓練資料中相似商品的銷售量作為特徵
* **各類型商品價格**
  * 商品單價較高的可能銷售量會較低，低單價的商品銷售量相對會比較高
* **儲存前處理數據**
  * 將前處理後的特徵數據另存為 pkl 檔案
  * 方便在訓練模型時不需要再花時間做資料前處理
* **每月銷售數量具有時間趨勢**
  * 延遲特徵考慮時間前後的關係，可能前 1 個月或前 2 個月和這個月的資料具有關聯性
  * 嘗試加入延遲 1、2、3、6 個月的特徵
* 加入延遲特徵前後的訓練結果：
                                                                                                             
                                                                                                             
<img src="https://github.com/hardychang/DSAI2022_Final-Predict-Future-Sales/blob/main/lag_table.png" width="400"/><br/>

## Model
### XGBoost
* Introduction
  * eta，在每次迭代更新中縮小特徵的權重避免模型 overfitting，default=0.3
  * max_depth，是樹狀模型的深度，增加深度可增加模型複雜度但是容易使模型 overfitting，default=6
  * min_child_weight，是模型中最小子節點權重的總和，在樹進行分區步驟時權重總和小於 min_child_weight，則不進行分區， min_child_weight 越大模型越可避免 overfitting，default=1
  * Subsample，避免模型 overfitting
  * tree_method='gpu_hist'，使用 GPU 版本
* Model Tuning
  * Grid Search CV 調整 XGBoost 的參數
  * max_depth=9、n_estimators=800、min_child_weight=400、 subsample=0.8 、colsample_bytree=0.8、eta=0.04、seed=40
* 參數調整前後的訓練結果：


<img src="https://github.com/hardychang/DSAI2022_Final-Predict-Future-Sales/blob/main/tuning_table.png" width="400"/><br/>

## Run

下載 dataset.zip、XGBModel.py、feature.py
將 dataset.zip 解壓縮後與 XGBModel.py、feature.py 儲存在同一路徑下
環境 Python "3.7.1"

```
conda create -n test python=="3.7"
```
```
activate test
```
安裝 requirements.txt 套件：
```
pip install -r requirements.txt
```
執行 feature.py 進行特徵擷取，會得到所有擷取後的特徵檔案 new_train.pkl
```
python feature.py
```
由於 feature.py 執行時間較長，可直接下載我們提供的[new_train.pkl](https://drive.google.com/file/d/1d4ftl5UkxkWifC570wUz3M17YrNkManh/view?usp=sharing)

將下載後的檔案存到與 XGBModel.py 的相同路徑下

接著執行 XGBModel.py 開始訓練 XGBoost 模型，最後得到 submission.csv 上傳至 Kaggle
```
python XGBModel.py
```
## Result
[submission.csv](https://github.com/hardychang/DSAI2022_Final-Predict-Future-Sales/blob/main/submission.csv)

2022/5/30 20:50 Score: 0.90605，Kaggle 排名：2995
## [Slides](https://docs.google.com/presentation/d/147nswLz32hbjwfVmo7aonrgVXcVC04X4/edit?usp=sharing&ouid=110596967971996559299&rtpof=true&sd=true)
