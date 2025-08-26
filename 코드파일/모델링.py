import pandas as pd
df=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_modify.csv',encoding='cp949')
df
df=df.drop(columns='성공여부')
df.info()

df['임대료(순이익)'] = pd.to_numeric(df['임대료(순이익)'], errors='coerce')

df=df.fillna(0)
df = df[df['매장당_평균_매출'] != 0]

df.info()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['매출안정성','업종과밀도(점포수)', '수요_float']] = scaler.fit_transform(
    df[['매출안정성','업종과밀도(점포수)','수요_float']]
)

df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_.csv')

df.info()
df.corr(numeric_only=True)['생존율'].sort_values(ascending=False)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rc('font', family='Malgun Gothic')
sns.boxplot(data=df, y='생존율')
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df.info()

X = df[['매출안정성','업종과밀도(점포수)','수요_float','임대료(순이익)','생존율']]

# 스케일링
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 클러스터별 평균값 확인
print(df.groupby('cluster').mean(numeric_only=True))
df.groupby('cluster')


df=df.drop(columns=['수요_category','매장당_평균_매출'])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df['임대료(순이익)']=scaler.fit_transform(df[['임대료(순이익)']])


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='Set1')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA + KMeans Cluster")
plt.legend()
plt.show()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# PCA 3차원
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)   # X_scaled = 스케일링된 데이터

# 3D 시각화
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:,0], X_pca[:,1], X_pca[:,2],
    c=df['cluster'], cmap='Set1', alpha=0.7
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3D) + KMeans Cluster")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()

df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_real_modify.csv')



################다시
df=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_modify.csv',encoding='cp949')
df=df.drop(columns=['수요_category', '매장당_평균_매출'])
df.info()
df['생존율']=scaler.fit_transform(df[['생존율']])
df
df['매출안정성']= -1 *df['매출안정성']
df['업종과밀도(점포수)']=-1*df['업종과밀도(점포수)']

df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_realreal_modify.csv')





import numpy as np
# 인코딩 오류 발생 시 아래 줄로 바꿔 실행 (cp949 or euc-kr)
# df = pd.read_csv(file_path, encoding="cp949")

print("데이터 로드 완료:", df.shape)
df.head()

# =========================
# 3. 기본 전처리
# =========================
# 3-1) Unnamed 컬럼 제거
unnamed_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
df = df.drop(columns=unnamed_cols, errors="ignore")

# 3-2) 숫자형 컬럼 변환
num_cols_candidates = ["매출안정성", "업종과밀도(점포수)", "수요_float", "생존율", "임대료(순이익)"]
for c in num_cols_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 3-3) 결측치 처리
# - 수치형 → 중앙값
# - 범주형 → "Unknown"
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]

for c in num_cols:
    if df[c].isna().any():
        med = df[c].median()
        df[c] = df[c].fillna(med)

for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna("Unknown")

print("결측치 처리 완료.")

# =========================
# 4. 이상치 처리 함수 정의
# =========================
def winsorize_clip(s: pd.Series, lower_q=0.01, upper_q=0.99):
    """윈저라이즈(quantile clipping)"""
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    clipped = s.clip(lower=lo, upper=hi)
    return clipped, lo, hi

def iqr_clip(s: pd.Series, k=1.5):
    """IQR 기반 클리핑"""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    clipped = s.clip(lower=lo, upper=hi)
    return clipped, lo, hi

def apply_clipping(df, col, method="winsor", **kwargs):
    before = df[col].copy()
    if method == "winsor":
        after, lo, hi = winsorize_clip(before, **kwargs)
    elif method == "iqr":
        after, lo, hi = iqr_clip(before, **kwargs)
    else:
        raise ValueError("method must be 'winsor' or 'iqr'")
    changed = (before != after).sum()
    df[col] = after
    return {"col": col, "method": method, "lower": lo, "upper": hi, "changed": int(changed)}

# =========================
# 5. 이상치 처리 적용
# =========================
report = []

if "임대료(순이익)" in df.columns:
    report.append(apply_clipping(df, "임대료(순이익)", method="winsor", lower_q=0.01, upper_q=0.99))
if "업종과밀도(점포수)" in df.columns:
    report.append(apply_clipping(df, "업종과밀도(점포수)", method="winsor", lower_q=0.01, upper_q=0.99))
if "매출안정성" in df.columns:
    report.append(apply_clipping(df, "매출안정성", method="iqr", k=1.5))
if "수요_float" in df.columns:
    report.append(apply_clipping(df, "수요_float", method="iqr", k=1.5))

print("이상치 처리 요약:")
print(pd.DataFrame(report))

# =========================
# 6. 이상치 처리 후 박스플롯 확인
# =========================
cols_to_plot = [c for c in ["임대료(순이익)", "업종과밀도(점포수)", "매출안정성", "수요_float"] if c in df.columns]
for c in cols_to_plot:
    plt.figure()
    df.boxplot(column=c)
    plt.title(f"Boxplot after clipping: {c}")
    plt.show()
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
# =========================
# 7. 저장
# =========================
save_path = "/content/final_data_real_modify_cleaned.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print("저장 완료:", save_path)


df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_realreal_modify.csv')



# KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 클러스터별 평균값 확인
print(df.groupby('cluster').mean(numeric_only=True))
df.groupby('cluster')


df=df.drop(columns=['수요_category','매장당_평균_매출'])

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df['임대료(순이익)']=scaler.fit_transform(df[['임대료(순이익)']])


from sklearn.decomposition import PCA


X = df[['매출안정성','업종과밀도(점포수)','수요_float','임대료(순이익)','생존율']]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster_knn_propagated'], cmap='Set1')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA + KMeans Cluster")
plt.legend()
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:,0], X_pca[:,1], X_pca[:,2],
    c=df['cluster_knn_propagated'], cmap='Set1', alpha=0.7
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3D) + KMeans Cluster")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()  

df=df.drop(columns='cluster')

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(X)
df

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=50)#핵심 심플:50개
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])





##다시
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_realreal진짜_modify.csv')

df.info()
# 1) 숫자 피처만 스케일링
num_cols = ['매출안정성','업종과밀도(점포수)','수요_float','생존율','임대료(순이익)']  # 필요 열만 남겨
X = df[num_cols].copy()


X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2) DBSCAN 클러스터링 (eps는 꼭 튜닝해라)
dbscan = DBSCAN(eps=0.9, min_samples=6, n_jobs=-1)
dbscan.fit(X_scaled)

labels = dbscan.labels_                 # 전체 포인트 라벨 (-1 = 노이즈)
core_idx = dbscan.core_sample_indices_  # 코어 샘플 인덱스
core_X = dbscan.components_             # 코어 샘플 좌표 (X_scaled[core_idx]와 동일)
core_y = labels[core_idx]               # 코어 샘플 라벨 (노이즈는 없음)

dbscan.components_

core_X.shape
# 3) 코어 샘플로 KNN 학습 (라벨 전파용)
#   - 코어 라벨 중 -1은 원래 없음. 혹시 섞였으면 필터링
core_X = dbscan.components_
core_y = labels[core_idx]
# DBSCAN의 코어샘플에는 -1 없음 → 따로 필터링할 필요 없다

knn = KNeighborsClassifier(n_neighbors=10)  # 50은 너무 클 수도. 코어 샘플 수 보고 줄여.
knn.fit(core_X, core_y)

# 4) 전체 포인트에 대해 "소프트" 라벨 전파 (원하면 노이즈만 예측)
pred_all = knn.predict(X_scaled)  # 모든 포인트에 예측 라벨 부여
# 노이즈만 덮어쓰고 싶으면:
propagated = labels.copy()
propagated[labels == -1] = pred_all[labels == -1]

# 5) 결과 붙이기
df['dbscan_label'] = labels
df['cluster_knn_propagated'] = propagated

df


from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# k = min_samples
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:,4])  # 4는 5번째 이웃
plt.plot(distances)
plt.ylabel("5-NN Distance")
plt.xlabel("Points sorted by distance")
plt.title("k-distance graph (choose elbow as eps)")
plt.show()



print(df.groupby('cluster_knn_propagated').mean(numeric_only=True))


###########Kmeans

df=df.drop(columns=['dbscan_label', 'cluster_knn_propagated'])
df

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[['매출안정성','업종과밀도(점포수)','수요_float','생존율','임대료(순이익)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

df['임대료(순이익)']=scaler.fit_transform(df[['임대료(순이익)']])
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

df
inertia = []
silhouette = []
K = range(2, 8)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)  # 클러스터 응집도
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(K, inertia, 'o-')
plt.xlabel("k")
plt.ylabel("Inertia (Within-cluster SSE)")
plt.title("엘보 방법")

plt.subplot(1,2,2)
plt.plot(K, silhouette, 'o-')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("실루엣 계수")

plt.tight_layout()
plt.show()

best_k = 4  # 엘보/실루엣 보고 골라
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X)   

print(df.groupby('cluster').mean(numeric_only=True))

df.describe()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df['cluster']=df['cluster'].replace({
    2:'유망',
    1:'비유망',
    0 : '비유망',
    3 : '비유망'
})
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['cluster'], cmap='Set1', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters (2D PCA)")
plt.show()

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




fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    X_pca[:,0], X_pca[:,1], X_pca[:,2       ],
    c=df['cluster'], cmap='Set1', alpha=0.7
)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("PCA (3D) + KMeans Cluster")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()  


df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/최종데이터셋/final_data_진짜임.csv')

df_filtered = df[(df['학교'] == '경희대') & (df['cluster'] == 2)]
print(df_filtered)

df_filtered=df_filtered.drop(columns=['매출안정성', '업종과밀도(점포수)','수요_float','생존율','임대료(순이익)','cluster'])

best_k = 2  # 엘보/실루엣 보고 골라
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)

X_filtered=df_filtered[['매출안정성','업종과밀도(점포수)', '수요_float','생존율','임대료(순이익)']]
df_filtered['cluster'] = kmeans.fit_predict(X_filtered)

df_filtered=df_filtered.drop(columns='cluster')
print(df_filtered.groupby('cluster').mean(numeric_only=True))

df.describe()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_filtered)

plt.scatter(X_pca[:,0], X_pca[:,1], c=df_filtered['cluster'], cmap='Set1', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("KMeans Clusters (2D PCA)")
plt.show()

df_filtered['cluster'].value_counts()

df_filtered[df_filtered['cluster']==2]




df_filtered['점수']=df['생존율']*0.5589 + df['수요_float']*0.207329+df['업종과밀도(점포수)']*0.120582+df['매출안정성']*0.065838+df['임대료(순이익)']*0.047652

df_filtered.sort_values(by='점수', ascending=False)

df_filtered_2=df_filtered.drop(columns=['매출안정성','업종과밀도(점포수)','수요_float','생존율','임대료(순이익)','cluster'])
df_filtered_2.sort_values(by='점수', ascending=False)