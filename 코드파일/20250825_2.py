import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# 니가 만든 클러스터별 평균 데이터프레임 예시
data = {
    'cluster': [0,1,2,3,4,5],
    '매출안정성': [-0.093457, 0.231188, 0.459509, 0.036984, 0.390281, 0.265508],
    '업종과밀도(점포수)': [-0.450261, 0.167077, -0.646512, -0.292100, 2.319272, -0.361888],
    '수요_float': [0.369010, -1.470845, -1.620362, 0.364005, 0.179569, 0.659854],
    '생존율': [-1.283280, -0.014141, -0.198224, 1.050031, 0.623610, -0.144391],
    '순이익': [-0.144130, -0.149367, 6.478260, -0.160877, -0.242835, -0.046745]
}
df_data = pd.DataFrame(data).set_index('cluster')

df_data


df_data['업종과밀도(점포수)']=df_data['업종과밀도(점포수)']*-1

df_data['최종점수']=df_data['매출안정성']*0.3 + df_data['업종과밀도(점포수)']*0.1 + df_data['수요_float']*0.1 + df_data['생존율']*0.4+df_data['순이익']*0.1
df_data.sort_values(by='최종점수',ascending=False)

