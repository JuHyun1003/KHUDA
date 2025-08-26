import pandas as pd

# 예시: 두 유사도 DataFrame (경희대 기준) 있다고 가정
# market_sim, demo_sim (index=학교, value=유사도 0~1)
market_sim = pd.Series({'경기대' : 0.944,'단국대':0.775,'명지대':0.869,'용인대':0.714,'성균관대':0.685,'수원대':0.659,'아주대' : 0.816})
demo_sim   = pd.Series({'경기대' : -0.284, '단국대':0.538,'명지대':0.508,'용인대':0.438,'성균관대':0.313,'수원대':-0.620,'아주대' : -0.698})

# 가중 평균
alpha = 0.3  # 상권 가중치
final_sim = alpha*market_sim + (1-alpha)*demo_sim

print(final_sim.sort_values(ascending=False))

