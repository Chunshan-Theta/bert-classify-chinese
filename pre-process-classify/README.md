### quick-start

#### 1. 將資料放在to_csv資料夾中（格式：<標籤> <內容>）

###### 無法提供完整訓練集 盡請見諒

```buildoutcfg
詢價	這個茶壺多少錢？
其他	你們營業時間是幾點
生氣	钱已经充到你们平台了，没金币玩屎呀
詢價	買這整組多少錢
詢價	買兩組的話多少
生氣	你再開玩笑嗎？
生氣	為什麼沒有打折？當初不是說只要800
生氣	你們都是骗子嗎
其他	妹仔多大啦
```

#### 2. 執行轉製腳本
```buildoutcfg
python3 count.py
```
#### 3. 產生檔案

* output_for_model/combined_chinese.csv
```buildoutcfg
問候	你好，有人在嗎？
詢價	這個茶壺多少錢？
其他	你們營業時間是幾點
生氣	钱已经充到你们平台了，没金币玩屎呀
詢價	買這整組多少錢
詢價	買兩組的話多少
生氣	你再開玩笑嗎？
生氣	為什麼沒有打折？當初不是說只要800
生氣	你們都是骗子嗎
其他	妹仔多大啦
問候	店員在嗎？
詢價	今日特價那組怎麼算
詢價	300元能買多少？
```
* output_for_model/for_train.csv

```buildoutcfg
0	你好，有人在嗎？
1	這個茶壺多少錢？
2	你們營業時間是幾點
3	钱已经充到你们平台了，没金币玩屎呀
1	買這整組多少錢
1	買兩組的話多少
3	你再開玩笑嗎？
3	為什麼沒有打折？當初不是說只要800
3	你們都是骗子嗎
2	妹仔多大啦
0	店員在嗎？
1	今日特價那組怎麼算
1	300元能買多少？
```