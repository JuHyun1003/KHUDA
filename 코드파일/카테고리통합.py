import pandas as pd
df=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/카드데이터/card_income_22_24.csv', encoding='cp949')

df['card_tpbuz_nm_1'].unique()

df.drop(df[df['card_tpbuz_nm_1'].isin(['공공/기업/단체', '공연/전시'])].index, inplace=True)
df.head()
df['card_tpbuz_nm_1']=df['card_tpbuz_nm_1'].replace('미디어/통신', '소매업')
df.head()
df['card_tpbuz_nm_2']=df['card_tpbuz_nm_2'].replace({
    '방송/미디어' : '전자상거래',
    '시스템/통신' : '전자상거래',
    '기타결제' : '전자상거래',
    '인터넷쇼핑' : '전자상거래'
})
df['card_tpbuz_nm_2']
df['card_tpbuz_nm_1']=df['card_tpbuz_nm_1'].replace('소매/유통', '소매업')
df['card_tpbuz_nm_2']=df['card_tpbuz_nm_2'].replace('제조/도매', '종합소매점')
df['card_tpbuz_nm_2'].value_counts()
df['card_tpbuz_nm_2']=df['card_tpbuz_nm_2'].replace('유아용품', '인테리어/가정용품')
df['card_tpbuz_nm_2']=df['card_tpbuz_nm_2'].replace({
    '광고/인쇄/인화' : '사무/교육용품',
    '차량판매' : '스포츠/레져용품',
    '무점포서비스' : '전자상거래',
    '사우나/휴게시설' : '숙박',
    '요가/단전/마사지' : '예체능계학원',
    '기술/직업교육학원' : '기타 학원',
    '기타교육' : '기타 학원',
    '유아교육' : '기타 학원',
    '자동차학원' : '기타 학원',
    '차량관리/부품' : '수리서비스',
    '연료판매' : '수리서비스',
    '고기요리' : '한식',
    '부페' : '한식',
    '일식/수산물' : '일식',
    '제과/제빵/떡/케익' : '제과점',
    '휴게소/대형업체' : '간이 음식업',
    '분식' : '간이 음식업',
    '간이주점' : '주점업',
    '유흥주점' : '주점업',
    '음식배달서비스' : '간이음식 포장전문점'
})
df['card_tpbuz_nm_2'].unique()
df['card_tpbuz_nm_1'].unique()

df['card_tpbuz_nm_2']=df['card_tpbuz_nm_2'].replace({
    '기타의료' : '의료업',
    '일반병원' : '의료업',
    '특화병원' : '의료업'
})

df.loc[df['card_tpbuz_nm_2']=='사무/교육용품', 'card_tpbuz_nm_1']='소매업'

df.loc[df['card_tpbuz_nm_2']=='전자상거래', 'card_tpbuz_nm_1']='소매업'

df.loc[df['card_tpbuz_nm_2']=='숙박박', 'card_tpbuz_nm_1']='서비스업'
df.loc[df['card_tpbuz_nm_2']=='예체능계학원', 'card_tpbuz_nm_1']='서비스업'
df.loc[df['card_tpbuz_nm_2']=='기타 학원', 'card_tpbuz_nm_1']='서비스업'
df.loc[df['card_tpbuz_nm_2']=='수리서비스', 'card_tpbuz_nm_1']='서비스업'

df['card_tpbuz_nm_1']=df['card_tpbuz_nm_1'].replace('음식', '음식업')

df.loc[df['card_tpbuz_nm_1']=='생활서비스', 'card_tpbuz_nm_2'].unique()

drop_list = [
    '가례서비스', '교통서비스', '렌탈서비스', 
    '보안/운송', '여행/유학대행', 
    '전문서비스', '회비/공과금', '금융상품/서비스'
]

df = df[~df['card_tpbuz_nm_2'].isin(drop_list)]

df['card_tpbuz_nm_1']=df['card_tpbuz_nm_1'].replace('생활서비스', '서비스업')

df.loc[:, 'card_tpbuz_nm_1'] = df['card_tpbuz_nm_1'].replace('생활서비스', '서비스업')
df['card_tpbuz_nm_1'].unique()

df.loc[df['card_tpbuz_nm_1']=='여가/오락', 'card_tpbuz_nm_2'].unique()

df.loc[
    (df['card_tpbuz_nm_1'] == '여가/오락') & 
    (df['card_tpbuz_nm_2'].isin(['숙박', '일반스포츠', '취미/오락'])),
    'card_tpbuz_nm_1'
] = '서비스업'


df.loc[df['card_tpbuz_nm_1']=='의료/건강', 'card_tpbuz_nm_2'].unique()

df = df.drop(
    df[(df['card_tpbuz_nm_1'] == '의료/건강') & 
       (df['card_tpbuz_nm_2'].isin(['수의업', '종합병원']))].index
)


df.loc[
    (df['card_tpbuz_nm_1'] == '의료/건강') & 
    (df['card_tpbuz_nm_2'].isin(['의료업'])),
    'card_tpbuz_nm_1'
] = '서비스업'

df.loc[
    (df['card_tpbuz_nm_1'] == '의료/건강') & 
    (df['card_tpbuz_nm_2'].isin(['의약/의료품'])),
    'card_tpbuz_nm_1'
] = '소매업'

df.loc[df['card_tpbuz_nm_1']=='학문/교육', 'card_tpbuz_nm_2'].unique()

df.loc[
    (df['card_tpbuz_nm_1'] == '학문/교육') & 
    (df['card_tpbuz_nm_2'].isin(['독서실/고시원','외국어학원','입시학원'])),
    'card_tpbuz_nm_1'
] = '서비스업'

df['card_tpbuz_nm_1'].value_counts()
df['card_tpbuz_nm_2'].unique()

df=df.drop(df[df['card_tpbuz_nm_2'].isin(['방문판매', '악기/공예', '기타용품'])].index)

df.to_csv('C:/portfolio/khuda_toyproject/데이터셋/카드데이터/ex1.csv')

# groupby + sum
df_grouped = df.groupby(['school', 'card_tpbuz_nm_1', 'card_tpbuz_nm_2']).sum(numeric_only=True).reset_index()
df_grouped.to_csv('C:/portfolio/khuda_toyproject/데이터셋/카드데이터/ex2.csv')

import pandas as pd
df=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/수요데이터/수요데이터.csv',encoding='cp949')
df.to_excel('C:/portfolio/khuda_toyproject/데이터셋/수요데이터/search_trend.xlsx')