#3.1 mnist
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
#fetch_openml() : 입력 : 판다스 데이터프레임, 레이블 : 판다스 시리즈로 받음. 그러나, mnist : 이미지이므로, as_Frame=False 사용
X, y=mnist.data, mnist.target
X   
y
y.shape

#시각화 함수 정의
import matplotlib.pyplot as plt
def plot_digit(image_data):
    image=image_data.reshape(28,28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')

#이미지 확인
some_digit=X[0]
plot_digit(some_digit)
plt.show()
#해당 이미지의 실제 정답
y[0]

#데이터 조사 전 테스트 세트 만들기
#앞쪽 60000개 : 훈련 세트
#뒤쪽 10000개 : 테스트 세트
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


#3.2 이진 분류기
#'5'와 '5 아님' 두 개의 클래스 구분

#타깃 벡터 제작
y_train_5=(y_train == '5')
y_test_5=(y_test == '5')

#분류모델(확률적 경사 하강법(SGD) 사용)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
#모델을 활용한 이미지 감지
sgd_clf.predict([some_digit])



#3.3 성능 측정
#3.3.1 교차 검증을 통한 정확도 측정
#cross_val_score()함수 사용하여 k-fold 교차 검증

#교차 검증 구현
#cross_val_score()함수와 거의 같은 작업, 동일한 결과 출력
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds=StratifiedKFold(n_splits=3) #데이터를 3등분해서 3번 훈련/검증

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_cif=clone(sgd_clf) #매 반복마다 기존 sgd_clf모델 복사해서 쓰기
    X_train_folds=X_train[train_index] #훈련 fold추출
    y_train_folds=y_train_5[train_index] #훈련 fold 추출
    X_test_fold=X_train[test_index] #검증fold 추출
    y_test_fold=y_train_5[test_index] #검증 fold 추출
    clone_cif.fit(X_train_folds, y_train_folds) #해당 fold 훈련 데이터로 학습
    y_pred=clone_cif.predict(X_test_fold) #검증 fold에 대해 예측
    n_correct = sum(y_pred==y_test_fold) #예측값과 실제값 비교해서 맞은 갯수 세기
    print(n_correct/len(y_pred)) #정확도(맞은갯수 / 전체)


from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

#더미 분류기 제작
#모든 이미지를 가장 많이 등장하는 클래스(5아님)으로 분류
from sklearn.dummy import DummyClassifier

dummy_cif = DummyClassifier()
dummy_cif.fit(X_train, y_train_5)
print(any(dummy_cif.predict(X_train))) #false 출력. True로 예측된 것이 없다

#모델의 정확도
cross_val_score(dummy_cif, X_train, y_train_5, cv=3, scoring='accuracy')


#3.3.2 오차 행렬
#정확도는 분류기의 성능 측정 지표로 선호하지 않음. 특히, 불균형한 데이터셋의 경우.
#오차행렬 : 분류기의 성능을 평가하는 더 좋은 방법

from sklearn.model_selection import cross_val_predict
#각 폴드마다 2개의 fold로 sgd_clf예측, 나머지 1개 fold에 대해 예측
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_train_5, y_train_pred)
cm
#| 실제↓ / 예측→  | 5 아님  | 5 맞음 |
#| ---------- | ----- | ---- |
#| 5 아님   | 53892(진짜 음성, TN) | 687(거짓 양성 / 1종 오류, FP)  |
#| 5 맞음   | 1891(거짓 음성, FN)  | 3530(진짜 양성, TP) |

#정밀도 : TP/TP+FP
#재현율 : TP/TP+FN

from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred) #3530/3530+686
# 5로 감지된 숫자 중 83.7%만 정확
recall_score(y_train_5, y_train_pred) #3530 / 3530 + 1891
#전체 숫자 5에서 65.1%만 감지

#F1 score : 정밀도와 재현율의 조화평균(정밀도, 재현율이 모두 높아야 F1값이 높아짐)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


#3.3.4 정밀도/재현율 트레이드오프
#정의 : 정밀도를 올리면 재현율이 줄고, 그 반대도 마찬가지
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold=0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred

threshold=3000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
#임곗값을 높이면 재현율이 줄어든다

#적절한 임곗값 정하기

#훈련 세트에 있는 모든 샘플의 점수 구하기
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
#가능한 모든 임곗값에 대해 정밀도, 재현율 계산
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
#정밀도, 재현율 시각화
import matplotlib
matplotlib.rc('font', family='Malgun Gothic')
matplotlib.rcParams['axes.unicode_minus'] = False
plt.plot(thresholds, precisions[:-1], 'b--', label='정밀도', linewidth=2)
plt.plot(thresholds, recalls[:-1], 'g-', label='재현율', linewidth=2)
plt.vlines(threshold, 0, 1.0, 'k', 'dotted', label='임곗값')
plt.legend()
plt.grid(True)
plt.show()
#재현율에 대한 정밀도 곡선
plt.plot(recalls, precisions, linewidth=2, label='정밀도/재현율 곡선')
[...]
plt.legend()
plt.grid(True)
plt.show() 
#정밀도 90% 달성하기! -> 정밀도가 최소 90%가 되는 가장 낮은 임곗값 찾기
idx_for_90_precision = (precisions >= 0.90).argmax()
threshold_for_90_precision=thresholds[idx_for_90_precision]
threshold_for_90_precision
#훈련 세트에 대한 예측
y_train_pred_90 = (y_scores >= threshold_for_90_precision)
precision_score(y_train_5, y_train_pred_90)

#3.3.5 ROC곡선
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

idx_for_thresholds_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_thresholds_at_90], fpr[idx_for_thresholds_at_90]
plt.plot(fpr, tpr, linewidth=2, label='ROC 곡선')
plt.plot([0,1], [0,1], 'k:', label='랜덤 분류기의 ROC 곡선')
plt.plot([fpr_90], [tpr_90], 'ko', label='90% 정밀도에 대한 임곗값')
plt.legend()
plt.grid(True)
plt.xlabel('거짓 양성 비율')
plt.ylabel('진짜 양성 비율(재현율)')
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)


#랜덤포레스트 만들어 SGDclassifier의 pr곡선과 f1비교
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
#RandomForestClassifier훈련
y_probas_forest=cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')

y_probas_forest[:2]

y_scores_forest=y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest)

#PR곡선
plt.plot(recalls_forest, precisions_forest, 'b-', linewidth=2, label='랜덤 포래스트')
plt.plot(recalls, precisions, '--', linewidth=2, label='SGD')
plt.legend()
plt.grid(True)
plt.show()



#3.4 다중 분류
from sklearn.svm import SVC
svm_clf=SVC(random_state=42)
svm_clf.fit(X_train[:2000], y_train[:2000])

svm_clf.predict([some_digit])

some_digit_scores=svm_clf.decision_function([some_digit])
some_digit_scores.round(2)
class_id=some_digit_scores.argmax()
class_id

svm_clf.classes_
svm_clf.classes_[class_id]

#사이킷런에서 OvO나 OvR을 사용하도록 강제
from sklearn.multiclass import OneVsOneClassifier
ovr_clf = OneVsOneClassifier(SVC(random_state=42))
ovr_clf.fit(X_train[:2000], y_train[:2000])
ovr_clf.predict([some_digit])
len(ovr_clf.estimators_)



import numpy as np
from sklearn.neighbors import KNeighborsClassifier

y_train_large=(y_train >= '7')
y_train_odd=(y_train.astype('int8') % 2==1)
y_multilabel=np.c_[y_train_large, y_train_odd]

knn_clf=KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

