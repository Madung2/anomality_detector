import pandas as pd
import wget

url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
file_path = wget.download(url, out='./data/KDDTrain+.txt')
print(file_path, '!!!!')