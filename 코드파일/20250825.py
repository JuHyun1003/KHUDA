import pandas as pd

df=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_realreal진짜_modify.csv')

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
df['임대료(순이익)']=scaler.fit_transform(df[['임대료(순이익)']])
df['업종과밀도(점포수)']=df['업종과밀도(점포수)']*-1

df
X = df[['매출안정성','업종과밀도(점포수)','수요_float','임대료(순이익)','생존율']]


kmeans = KMeans(n_clusters=6, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

print(df.groupby('cluster').mean(numeric_only=True))


import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('font', family='Malgun Gothic')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='Set1')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA + KMeans Cluster")
plt.legend()
plt.show()

df['cluster'].value_counts()

df[(df['cluster']==2)]

df['cluster'].value_counts()

df['cluster']=df['cluster'].replace({
    3 : '유망',
    0 : '비유망',
    1 : '비유망',
    2 : '비유망',
    4 : '비유망',
    5 : '비유망'
})

df['cluster'].value_counts()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1) 피처(X)와 타깃(y) 나누기
X = df[['매출안정성','업종과밀도(점포수)','수요_float','생존율','임대료(순이익)']]
y = df['cluster']   # 이미 '유망' / '비유망'으로 치환된 상태

# 2) 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) 랜덤포레스트 모델 학습
rf = RandomForestClassifier(
    n_estimators=200,   # 트리 개수
    max_depth=None,     # 트리 깊이 제한 없음
    random_state=42,
    class_weight='balanced'  # 비유망/유망 비율 불균형 있을 때
)
rf.fit(X_train, y_train)

# 4) 예측
y_pred = rf.predict(X_test)

# 5) 평가
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6) 변수 중요도 확인
import pandas as pd
import matplotlib.pyplot as plt

feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp.sort_values().plot(kind='barh', figsize=(6,4))
plt.title("Feature Importance (Random Forest)")
plt.show()

import pandas as pd

# 변수 중요도 추출
importances = rf.feature_importances_

# DataFrame으로 정리
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feat_imp_df)

df_grouped = df[(df['학교']=='경희대') & (df['cluster']==3)]

df_grouped=df_grouped.drop(columns='cluster')
df_grouped['최종점수']=df_grouped['매출안정성']*0.046606+df_grouped['임대료(순이익)']*0.068581+df_grouped['업종과밀도(점포수)']*0.087863+df_grouped['수요_float']*0.154372+df_grouped['생존율']*0.642578

df_grouped.sort_values(by='최종점수',ascending=False)

df_grouped=df_grouped.drop(columns=['매출안정성','업종과밀도(점포수)','수요_float','생존율','임대료(순이익)'])

