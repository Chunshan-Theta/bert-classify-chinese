import csv
from os import listdir
from os.path import isfile, join

data_dir = "to_csv"
source_files = [f for f in listdir('./{}/'.format(data_dir)) if isfile(join('./{}/'.format(data_dir), f))]
file_names = []

# 蒐集來源
for sf in source_files:
    #skip file if the file is not a csv file
    if not sf[-3:] == "csv": continue

    #skip output file from the script
    if sf == "combined.csv": continue

    file_names.append(sf)

rows_storage_chinese = []
rows_storage_index = []
count_row_type = {}
label_index = []
for f in file_names:
    # 開啟 CSV 檔案
    with open(data_dir+"/"+f) as csvfile:

      # 讀取CSV檔案內容
      rows = csv.reader(csvfile, delimiter='\t')

      # 以迴圈輸出每一列
      for row in rows:
        chinese_label = row[0]
        chinese_content = row[1]


        # 計算標籤數量
        if not chinese_label in count_row_type:
            count_row_type[chinese_label] = 1
        else:
            count_row_type[chinese_label] += 1

        # 改編標籤成編號
        if not chinese_label in label_index:
            label_index.append(chinese_label)

        # add to storage
        rows_storage_chinese.append([chinese_label, chinese_content])
        rows_storage_index.append([label_index.index(chinese_label), chinese_content])


# OUTPUT
with open("./output_for_model/combined_chinese.csv","w") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for r in rows_storage_chinese:
        spamwriter.writerow(r)
with open("./output_for_model/for_train.csv", "w") as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    for r in rows_storage_index:
        spamwriter.writerow(r)

    
print(label_index, count_row_type)