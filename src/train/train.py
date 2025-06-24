import pandas as pd
import numpy as np


# 1. 데이터 불러오기
file_path = './data/KDDTrain+.txt'

columns = [  # 41 features + label
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
    'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
    'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
    'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
    'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

# 2. 데이터 불러오기
df = pd.read_csv(file_path, names=columns)

# 3. 라벨 이진화
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# 4. 인코딩 & 스케일링
from sklearn.preprocessing import StandardScaler, LabelEncoder

for col in ['protocol_type', 'service', 'flag']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 특성과 라벨 분리
x = df.drop('label', axis=1)
y = df['label']

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# 5. 모델 학습
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
