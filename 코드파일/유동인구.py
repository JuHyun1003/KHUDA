import pandas as pd
df_suwon=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_수원시_202407.csv')
df_suwon_8=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_수원시_202408.csv')
df_suwon_9=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_수원시_202409.csv')
df_suwon_10=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_수원시_202410.csv', encoding='cp949')
df_suwon_11=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_수원시_202411.csv')
df_suwon_12=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_수원시_202412.csv', encoding='cp949')
df_suwon_1=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_202501_수원시.csv',encoding='cp949')
df_suwon_2=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_202502_수원시.csv')

df_suwon_1.columns = df_suwon_1.columns.str.strip()
df_suwon_2.columns = df_suwon_2.columns.str.strip()

df_suwon_1.info()
df_suwon_1.head()
df_suwon=df_suwon.drop(columns=['ETL_YMD'])
df_suwon_8=df_suwon_8.drop(columns=['ETL_YMD'])
df_suwon_9=df_suwon_9.drop(columns=['ETL_YMD'])
df_suwon_10=df_suwon_10.drop(columns=['ETL_YMD'])
df_suwon_11=df_suwon_11.drop(columns=['ETL_YMD'])
df_suwon_12=df_suwon_12.drop(columns=['ETL_YMD'])
df_suwon_1=df_suwon_1.drop(columns=['ETL_YMD'])
df_suwon_2=df_suwon_2.drop(columns=['ETL_YMD'])

def make_suwon_kyunghee(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '영통3동') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_kyunghee_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_suwon_kyunghee_11=make_suwon_kyunghee(df_suwon_11,11)
df_suwon_kyunghee_11
df_suwon_kyunghee_12=make_suwon_kyunghee(df_suwon_12,12)
df_suwon_kyunghee_12
df_suwon_kyunghee_1=make_suwon_kyunghee(df_suwon_1, 1)
df_suwon_kyunghee_1
df_suwon_kyunghee_2=make_suwon_kyunghee(df_suwon_2, 2)
df_suwon_kyunghee_2


##경희대 7월
df_suwon_kyunghee_7 = df_suwon[df_suwon['ADMI_NM'] == '영통3동']
df_suwon_kyunghee_7 = df_suwon_kyunghee_7[df_suwon['FORN_GB'] == '내국인']
df_suwon_kyunghee_7


col_sum_kyunghee_7 = df_suwon_kyunghee_7.sum(numeric_only=True)
df_suwon_kyunghee_7.loc['col_sum_kyunghee_7'] = col_sum_kyunghee_7

df_suwon_kyunghee_7 = df_suwon_kyunghee_7.loc[['col_sum_kyunghee_7']]
df_suwon_kyunghee_7 = df_suwon_kyunghee_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_kyunghee_7
##경희대 8월
df_suwon_kyunghee_8 = df_suwon_8[df_suwon_8['ADMI_NM'] == '영통3동']
df_suwon_kyunghee_8 = df_suwon_kyunghee_8[df_suwon_8['FORN_GB'] == '내국인']
df_suwon_kyunghee_8


col_sum_kyunghee_8 = df_suwon_kyunghee_8.sum(numeric_only=True)
df_suwon_kyunghee_8.loc['col_sum_kyunghee_8'] = col_sum_kyunghee_8

df_suwon_kyunghee_8 = df_suwon_kyunghee_8.loc[['col_sum_kyunghee_8']]
df_suwon_kyunghee_8 = df_suwon_kyunghee_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_kyunghee_8
##경희대 9월
df_suwon_kyunghee_9 = df_suwon_9[df_suwon_9['ADMI_NM'] == '영통3동']
df_suwon_kyunghee_9 = df_suwon_kyunghee_9[df_suwon_9['FORN_GB'] == '내국인']
df_suwon_kyunghee_9


col_sum_kyunghee_9 = df_suwon_kyunghee_9.sum(numeric_only=True)
df_suwon_kyunghee_9.loc['col_sum_kyunghee_9'] = col_sum_kyunghee_9

df_suwon_kyunghee_9 = df_suwon_kyunghee_9.loc[['col_sum_kyunghee_9']]
df_suwon_kyunghee_9 = df_suwon_kyunghee_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_kyunghee_9
##경희대 10월
df_suwon_kyunghee_10 = df_suwon_10[df_suwon_10['ADMI_NM'] == '영통3동']
df_suwon_kyunghee_10 = df_suwon_kyunghee_10[df_suwon_10['FORN_GB'] == '내국인']
df_suwon_kyunghee_10


col_sum_kyunghee_10 = df_suwon_kyunghee_10.sum(numeric_only=True)
df_suwon_kyunghee_10.loc['col_sum_kyunghee_10'] = col_sum_kyunghee_10

df_suwon_kyunghee_10 = df_suwon_kyunghee_10.loc[['col_sum_kyunghee_10']]
df_suwon_kyunghee_10 = df_suwon_kyunghee_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_kyunghee_10

##경희대 7~10월
df_kyunghee_1_12 = pd.concat(
    [df_suwon_kyunghee_1,
     df_suwon_kyunghee_2,
     df_suwon_kyunghee_7,
     df_suwon_kyunghee_8,
     df_suwon_kyunghee_9,
     df_suwon_kyunghee_10,
     df_suwon_kyunghee_11,
     df_suwon_kyunghee_12],
    axis=0   # 행 기준 합치기
)
df_kyunghee_1_12


def make_suwon_ajou(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '원천동') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_ajou_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_suwon_ajou_11=make_suwon_ajou(df_suwon_11, 11)
df_suwon_ajou_11
df_suwon_ajou_12=make_suwon_ajou(df_suwon_12,12)
df_suwon_ajou_12
df_suwon_ajou_1=make_suwon_ajou(df_suwon_1,1)
df_suwon_ajou_1
df_suwon_ajou_2=make_suwon_ajou(df_suwon_2,2)
df_suwon_ajou_2
##아주대 7월
df_suwon_ajou_7 = df_suwon[df_suwon['ADMI_NM'] == '원천동']
df_suwon_ajou_7 = df_suwon_ajou_7[df_suwon['FORN_GB'] == '내국인']
df_suwon_ajou_7

col_sum_ajou_7 = df_suwon_ajou_7.sum(numeric_only=True)
df_suwon_ajou_7.loc['col_sum_ajou_7'] = col_sum_ajou_7

df_suwon_ajou_7 = df_suwon_ajou_7.loc[['col_sum_ajou_7']]
df_suwon_ajou_7 = df_suwon_ajou_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_ajou_7

##아주대 8월
df_suwon_ajou_8 = df_suwon_8[df_suwon_8['ADMI_NM'] == '원천동']
df_suwon_ajou_8 = df_suwon_ajou_8[df_suwon_8['FORN_GB'] == '내국인']
df_suwon_ajou_8

col_sum_ajou_8 = df_suwon_ajou_8.sum(numeric_only=True)
df_suwon_ajou_8.loc['col_sum_ajou_8'] = col_sum_ajou_8

df_suwon_ajou_8 = df_suwon_ajou_8.loc[['col_sum_ajou_8']]
df_suwon_ajou_8 = df_suwon_ajou_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_ajou_8

##아주대 9월
df_suwon_ajou_9 = df_suwon_9[df_suwon_9['ADMI_NM'] == '원천동']
df_suwon_ajou_9 = df_suwon_ajou_9[df_suwon_9['FORN_GB'] == '내국인']
df_suwon_ajou_9

col_sum_ajou_9 = df_suwon_ajou_9.sum(numeric_only=True)
df_suwon_ajou_9.loc['col_sum_ajou_9'] = col_sum_ajou_9

df_suwon_ajou_9 = df_suwon_ajou_9.loc[['col_sum_ajou_9']]
df_suwon_ajou_9 = df_suwon_ajou_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_ajou_9

##아주대 10월
df_suwon_ajou_10 = df_suwon_10[df_suwon_10['ADMI_NM'] == '원천동']
df_suwon_ajou_10 = df_suwon_ajou_10[df_suwon_10['FORN_GB'] == '내국인']
df_suwon_ajou_10

col_sum_ajou_10 = df_suwon_ajou_10.sum(numeric_only=True)
df_suwon_ajou_10.loc['col_sum_ajou_10'] = col_sum_ajou_10

df_suwon_ajou_10 = df_suwon_ajou_10.loc[['col_sum_ajou_10']]
df_suwon_ajou_10 = df_suwon_ajou_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_ajou_10

##아주대 7~10월
df_ajou_1_12 = pd.concat(
    [df_suwon_ajou_1,
     df_suwon_ajou_2,
     df_suwon_ajou_7,
     df_suwon_ajou_8,
     df_suwon_ajou_9,
     df_suwon_ajou_10,
     df_suwon_ajou_11,
     df_suwon_ajou_12],
    axis=0   # 행 기준 합치기
)
df_ajou_1_12



def make_suwon_gyeonggi(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '광교1동') & (df['FORN_GB'] == '내국인')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_gyeonggi_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_suwon_gyeonggi_11=make_suwon_gyeonggi(df_suwon_11,11)
df_suwon_gyeonggi_12=make_suwon_gyeonggi(df_suwon_12,12)
df_suwon_gyeonggi_1=make_suwon_gyeonggi(df_suwon_1,1)
df_suwon_gyeonggi_2=make_suwon_gyeonggi(df_suwon_2,2)
##경기대 7월
df_suwon_gyeonggi_7 = df_suwon[df_suwon['ADMI_NM'] == '광교1동']
df_suwon_gyeonggi_7 = df_suwon_gyeonggi_7[df_suwon['FORN_GB'] == '내국인']
df_suwon_gyeonggi_7

col_sum_gyeonggi_7 = df_suwon_gyeonggi_7.sum(numeric_only=True)
df_suwon_gyeonggi_7.loc['col_sum_gyeonggi_7'] = col_sum_gyeonggi_7

df_suwon_gyeonggi_7 = df_suwon_gyeonggi_7.loc[['col_sum_gyeonggi_7']]
df_suwon_gyeonggi_7 = df_suwon_gyeonggi_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_gyeonggi_7

##경기대 8월
df_suwon_gyeonggi_8 = df_suwon_8[df_suwon_8['ADMI_NM'] == '광교1동']
df_suwon_gyeonggi_8 = df_suwon_gyeonggi_8[df_suwon_8['FORN_GB'] == '내국인']
df_suwon_gyeonggi_8

col_sum_gyeonggi_8 = df_suwon_gyeonggi_8.sum(numeric_only=True)
df_suwon_gyeonggi_8.loc['col_sum_gyeonggi_8'] = col_sum_gyeonggi_8

df_suwon_gyeonggi_8 = df_suwon_gyeonggi_8.loc[['col_sum_gyeonggi_8']]
df_suwon_gyeonggi_8 = df_suwon_gyeonggi_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_gyeonggi_8

##경기대 9월
df_suwon_gyeonggi_9 = df_suwon_9[df_suwon_9['ADMI_NM'] == '광교1동']
df_suwon_gyeonggi_9 = df_suwon_gyeonggi_9[df_suwon_9['FORN_GB'] == '내국인']
df_suwon_gyeonggi_9

col_sum_gyeonggi_9 = df_suwon_gyeonggi_9.sum(numeric_only=True)
df_suwon_gyeonggi_9.loc['col_sum_gyeonggi_9'] = col_sum_gyeonggi_9

df_suwon_gyeonggi_9 = df_suwon_gyeonggi_9.loc[['col_sum_gyeonggi_9']]
df_suwon_gyeonggi_9 = df_suwon_gyeonggi_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_gyeonggi_9

##경기대 10월
df_suwon_gyeonggi_10 = df_suwon_10[df_suwon_10['ADMI_NM'] == '광교1동']
df_suwon_gyeonggi_10 = df_suwon_gyeonggi_10[df_suwon_10['FORN_GB'] == '내국인']
df_suwon_gyeonggi_10

col_sum_gyeonggi_10 = df_suwon_gyeonggi_10.sum(numeric_only=True)
df_suwon_gyeonggi_10.loc['col_sum_gyeonggi_10'] = col_sum_gyeonggi_10

df_suwon_gyeonggi_10 = df_suwon_gyeonggi_10.loc[['col_sum_gyeonggi_10']]
df_suwon_gyeonggi_10 = df_suwon_gyeonggi_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_gyeonggi_10

##경기대 7~10월
df_gyeonggi_1_12 = pd.concat(
    [df_suwon_gyeonggi_1,
     df_suwon_gyeonggi_2,
     df_suwon_gyeonggi_7,
     df_suwon_gyeonggi_8,
     df_suwon_gyeonggi_9,
     df_suwon_gyeonggi_10,
     df_suwon_gyeonggi_11,
     df_suwon_gyeonggi_12],
    axis=0   # 행 기준 합치기
)
df_gyeonggi_1_12

def make_suwon_skku(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '율천동') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_skku_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_suwon_skku_11=make_suwon_skku(df_suwon_11,11)
df_suwon_skku_12=make_suwon_skku(df_suwon_12,12)
df_suwon_skku_1=make_suwon_skku(df_suwon_1,1)
df_suwon_skku_2=make_suwon_skku(df_suwon_2,2)
##성대7월
df_suwon_skku_7 = df_suwon[df_suwon['ADMI_NM'] == '율천동']
df_suwon_skku_7 = df_suwon_skku_7[df_suwon['FORN_GB'] == '내국인']
df_suwon_skku_7

col_sum_skku_7 = df_suwon_skku_7.sum(numeric_only=True)
df_suwon_skku_7.loc['col_sum_skku_7'] = col_sum_skku_7

df_suwon_skku_7 = df_suwon_skku_7.loc[['col_sum_skku_7']]
df_suwon_skku_7 = df_suwon_skku_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_skku_7

##성대8월
df_suwon_skku_8 = df_suwon_8[df_suwon_8['ADMI_NM'] == '율천동']
df_suwon_skku_8 = df_suwon_skku_8[df_suwon_8['FORN_GB'] == '내국인']
df_suwon_skku_8

col_sum_skku_8 = df_suwon_skku_8.sum(numeric_only=True)
df_suwon_skku_8.loc['col_sum_skku_8'] = col_sum_skku_8

df_suwon_skku_8 = df_suwon_skku_8.loc[['col_sum_skku_8']]
df_suwon_skku_8 = df_suwon_skku_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_skku_8

##성대9월
df_suwon_skku_9 = df_suwon_9[df_suwon_9['ADMI_NM'] == '율천동']
df_suwon_skku_9 = df_suwon_skku_9[df_suwon_9['FORN_GB'] == '내국인']
df_suwon_skku_9

col_sum_skku_9 = df_suwon_skku_9.sum(numeric_only=True)
df_suwon_skku_9.loc['col_sum_skku_9'] = col_sum_skku_9

df_suwon_skku_9 = df_suwon_skku_9.loc[['col_sum_skku_9']]
df_suwon_skku_9 = df_suwon_skku_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_skku_9

##성대10월
df_suwon_skku_10 = df_suwon_10[df_suwon_10['ADMI_NM'] == '율천동']
df_suwon_skku_10 = df_suwon_skku_10[df_suwon_10['FORN_GB'] == '내국인']
df_suwon_skku_10

col_sum_skku_10 = df_suwon_skku_10.sum(numeric_only=True)
df_suwon_skku_10.loc['col_sum_skku_10'] = col_sum_skku_10

df_suwon_skku_10 = df_suwon_skku_10.loc[['col_sum_skku_10']]
df_suwon_skku_10 = df_suwon_skku_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_suwon_skku_10

##성균관대 7~10월
df_skku_1_12 = pd.concat(
    [df_suwon_skku_1,
     df_suwon_skku_2,
     df_suwon_skku_7,
     df_suwon_skku_8,
     df_suwon_skku_9,
     df_suwon_skku_10,
     df_suwon_skku_11,
     df_suwon_skku_12],
    axis=0   # 행 기준 합치기
)
df_skku_1_12


##용인관할대학
df_yongin_7=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_용인시_202407.csv')
df_yongin_8=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_용인시_202408.csv')
df_yongin_9=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_용인시_202409.csv')
df_yongin_10=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_용인시_202410.csv',encoding='cp949')
df_yongin_11=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_용인시_202411.csv')
df_yongin_12=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_용인시_202412.csv', encoding='cp949')
df_yongin_1=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_202501_용인시.csv')
df_yongin_2=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_202502_용인시.csv')

df_yongin_1.columns = df_yongin_1.columns.str.strip()
df_yongin_2.columns = df_yongin_2.columns.str.strip()

df_yongin_1=df_yongin_1.drop(columns=['ETL_YMD'])
df_yongin_2=df_yongin_2.drop(columns=['ETL_YMD'])
df_yongin_7=df_yongin_7.drop(columns=['ETL_YMD'])
df_yongin_8=df_yongin_8.drop(columns=['ETL_YMD'])
df_yongin_9=df_yongin_9.drop(columns=['ETL_YMD'])
df_yongin_10=df_yongin_10.drop(columns=['ETL_YMD'])
df_yongin_11=df_yongin_11.drop(columns=['ETL_YMD'])
df_yongin_12=df_yongin_12.drop(columns=['ETL_YMD'])


def make_yongin_dku(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '죽전3동') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_dku_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_yongin_dku_11=make_yongin_dku(df_yongin_11,11)
df_yongin_dku_12=make_yongin_dku(df_yongin_12,12)
df_yongin_dku_1=make_yongin_dku(df_yongin_1,1)
df_yongin_dku_2=make_yongin_dku(df_yongin_2,2)
##단국대7월
df_yongin_dku_7 = df_yongin_7[df_yongin_7['ADMI_NM'] == '죽전3동']
df_yongin_dku_7 = df_yongin_dku_7[df_yongin_7['FORN_GB'] == '내국인']
df_yongin_dku_7

col_sum_dku_7 = df_yongin_dku_7.sum(numeric_only=True)
df_yongin_dku_7.loc['col_sum_dku_7'] = col_sum_dku_7

df_yongin_dku_7 = df_yongin_dku_7.loc[['col_sum_dku_7']]
df_yongin_dku_7 = df_yongin_dku_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_dku_7

##단국대8월
df_yongin_dku_8 = df_yongin_8[df_yongin_8['ADMI_NM'] == '죽전3동']
df_yongin_dku_8 = df_yongin_dku_8[df_yongin_8['FORN_GB'] == '내국인']
df_yongin_dku_8

col_sum_dku_8 = df_yongin_dku_8.sum(numeric_only=True)
df_yongin_dku_8.loc['col_sum_dku_8'] = col_sum_dku_8

df_yongin_dku_8 = df_yongin_dku_8.loc[['col_sum_dku_8']]
df_yongin_dku_8 = df_yongin_dku_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_dku_8

##단국대9월
df_yongin_dku_9 = df_yongin_9[df_yongin_9['ADMI_NM'] == '죽전3동']
df_yongin_dku_9 = df_yongin_dku_9[df_yongin_9['FORN_GB'] == '내국인']
df_yongin_dku_9

col_sum_dku_9 = df_yongin_dku_9.sum(numeric_only=True)
df_yongin_dku_9.loc['col_sum_dku_9'] = col_sum_dku_9

df_yongin_dku_9 = df_yongin_dku_9.loc[['col_sum_dku_9']]
df_yongin_dku_9 = df_yongin_dku_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_dku_9

##단국대10월
df_yongin_dku_10 = df_yongin_10[df_yongin_10['ADMI_NM'] == '죽전3동']
df_yongin_dku_10 = df_yongin_dku_10[df_yongin_10['FORN_GB'] == '내국인']
df_yongin_dku_10

col_sum_dku_10 = df_yongin_dku_10.sum(numeric_only=True)
df_yongin_dku_10.loc['col_sum_dku_10'] = col_sum_dku_10

df_yongin_dku_10 = df_yongin_dku_10.loc[['col_sum_dku_10']]
df_yongin_dku_10 = df_yongin_dku_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_dku_10

df_dku_1_12 = pd.concat(
    [df_yongin_dku_1,
     df_yongin_dku_2,
     df_yongin_dku_7,
     df_yongin_dku_8,
     df_yongin_dku_9,
     df_yongin_dku_10,
     df_yongin_dku_11,
     df_yongin_dku_12],
    axis=0   # 행 기준 합치기
)
df_dku_1_12


def make_yongin_myeongji(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '역북동') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_myeongji_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_yongin_myeongji_11=make_yongin_myeongji(df_yongin_11,11)
df_yongin_myeongji_12=make_yongin_myeongji(df_yongin_12,12)
df_yongin_myeongji_1=make_yongin_myeongji(df_yongin_1,1)
df_yongin_myeongji_2=make_yongin_myeongji(df_yongin_2,2)

##명지대 7월
df_yongin_myeongji_7 = df_yongin_7[df_yongin_7['ADMI_NM'] == '역북동']
df_yongin_myeongji_7 = df_yongin_myeongji_7[df_yongin_7['FORN_GB'] == '내국인']
df_yongin_myeongji_7

col_sum_myeongji_7 = df_yongin_myeongji_7.sum(numeric_only=True)
df_yongin_myeongji_7.loc['col_sum_myeongji_7'] = col_sum_myeongji_7

df_yongin_myeongji_7 = df_yongin_myeongji_7.loc[['col_sum_myeongji_7']]
df_yongin_myeongji_7 = df_yongin_myeongji_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_myeongji_7

##명지대 8월
df_yongin_myeongji_8 = df_yongin_8[df_yongin_8['ADMI_NM'] == '역북동']
df_yongin_myeongji_8 = df_yongin_myeongji_8[df_yongin_8['FORN_GB'] == '내국인']
df_yongin_myeongji_8

col_sum_myeongji_8 = df_yongin_myeongji_8.sum(numeric_only=True)
df_yongin_myeongji_8.loc['col_sum_myeongji_8'] = col_sum_myeongji_8

df_yongin_myeongji_8 = df_yongin_myeongji_8.loc[['col_sum_myeongji_8']]
df_yongin_myeongji_8 = df_yongin_myeongji_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_myeongji_8

##명지대 9월
df_yongin_myeongji_9 = df_yongin_9[df_yongin_9['ADMI_NM'] == '역북동']
df_yongin_myeongji_9 = df_yongin_myeongji_9[df_yongin_9['FORN_GB'] == '내국인']
df_yongin_myeongji_9

col_sum_myeongji_9 = df_yongin_myeongji_9.sum(numeric_only=True)
df_yongin_myeongji_9.loc['col_sum_myeongji_9'] = col_sum_myeongji_9

df_yongin_myeongji_9 = df_yongin_myeongji_9.loc[['col_sum_myeongji_9']]
df_yongin_myeongji_9 = df_yongin_myeongji_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_myeongji_9

##명지대 10월
df_yongin_myeongji_10 = df_yongin_10[df_yongin_10['ADMI_NM'] == '역북동']
df_yongin_myeongji_10 = df_yongin_myeongji_10[df_yongin_10['FORN_GB'] == '내국인']
df_yongin_myeongji_10

col_sum_myeongji_10 = df_yongin_myeongji_10.sum(numeric_only=True)
df_yongin_myeongji_10.loc['col_sum_myeongji_10'] = col_sum_myeongji_10

df_yongin_myeongji_10 = df_yongin_myeongji_10.loc[['col_sum_myeongji_10']]
df_yongin_myeongji_10 = df_yongin_myeongji_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_myeongji_10

df_myeongji_1_12 = pd.concat(
    [df_yongin_myeongji_1,
     df_yongin_myeongji_2,
     df_yongin_myeongji_7,
     df_yongin_myeongji_8,
     df_yongin_myeongji_9,
     df_yongin_myeongji_10,
     df_yongin_myeongji_11,
     df_yongin_myeongji_12],
    axis=0   # 행 기준 합치기
)
df_myeongji_1_12


def make_yongin_yu(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '삼가동') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_yu_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp
df_yongin_yu_11=make_yongin_yu(df_yongin_11,11)
df_yongin_yu_12=make_yongin_yu(df_yongin_12,12)
df_yongin_yu_1=make_yongin_yu(df_yongin_1,1)
df_yongin_yu_2=make_yongin_yu(df_yongin_2,2)

##용인대7월
df_yongin_yu_7 = df_yongin_7[df_yongin_7['ADMI_NM'] == '삼가동']
df_yongin_yu_7 = df_yongin_yu_7[df_yongin_7['FORN_GB'] == '내국인']
df_yongin_yu_7

col_sum_yu_7 = df_yongin_yu_7.sum(numeric_only=True)
df_yongin_yu_7.loc['col_sum_yu_7'] = col_sum_yu_7

df_yongin_yu_7 = df_yongin_yu_7.loc[['col_sum_yu_7']]
df_yongin_yu_7 = df_yongin_yu_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_yu_7

##용인대8월
df_yongin_yu_8 = df_yongin_8[df_yongin_8['ADMI_NM'] == '삼가동']
df_yongin_yu_8 = df_yongin_yu_8[df_yongin_8['FORN_GB'] == '내국인']
df_yongin_yu_8

col_sum_yu_8 = df_yongin_yu_8.sum(numeric_only=True)
df_yongin_yu_8.loc['col_sum_yu_8'] = col_sum_yu_8

df_yongin_yu_8 = df_yongin_yu_8.loc[['col_sum_yu_8']]
df_yongin_yu_8 = df_yongin_yu_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_yu_8

##용인대9월
df_yongin_yu_9 = df_yongin_9[df_yongin_9['ADMI_NM'] == '삼가동']
df_yongin_yu_9 = df_yongin_yu_9[df_yongin_9['FORN_GB'] == '내국인']
df_yongin_yu_9

col_sum_yu_9 = df_yongin_yu_9.sum(numeric_only=True)
df_yongin_yu_9.loc['col_sum_yu_9'] = col_sum_yu_9

df_yongin_yu_9 = df_yongin_yu_9.loc[['col_sum_yu_9']]
df_yongin_yu_9 = df_yongin_yu_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_yu_9

##용인대10월
df_yongin_yu_10 = df_yongin_10[df_yongin_10['ADMI_NM'] == '삼가동']
df_yongin_yu_10 = df_yongin_yu_10[df_yongin_10['FORN_GB'] == '내국인']
df_yongin_yu_10

col_sum_yu_10 = df_yongin_yu_10.sum(numeric_only=True)
df_yongin_yu_10.loc['col_sum_yu_10'] = col_sum_yu_10

df_yongin_yu_10 = df_yongin_yu_10.loc[['col_sum_yu_10']]
df_yongin_yu_10 = df_yongin_yu_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_yongin_yu_10

df_yu_1_12 = pd.concat(
    [df_yongin_yu_1,
     df_yongin_yu_2,
     df_yongin_yu_7,
     df_yongin_yu_8,
     df_yongin_yu_9,
     df_yongin_yu_10,
     df_yongin_yu_11,
     df_yongin_yu_12],
    axis=0   # 행 기준 합치기
)
df_yu_1_12

##화성시
df_hwasung_7=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_화성시_202407.csv')
df_hwasung_8=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_화성시_202408.csv')
df_hwasung_9=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_화성시_202409.csv')
df_hwasung_10=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_화성시_202410.csv' ,encoding='cp949')
df_hwasung_11=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_화성시_202411.csv')
df_hwasung_12=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_화성시_202412.csv',encoding='cp949')
df_hwasung_1=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_202501_화성시.csv')
df_hwasung_2=pd.read_csv('C:/portfolio/khuda_toyproject/데이터셋/유동인구데이터/T22_GG_ADMI_FLOWPOP_202502_화성시.csv')


df_hwasung_1.columns = df_hwasung_1.columns.str.strip()
df_hwasung_2.columns = df_hwasung_2.columns.str.strip()


df_hwasung_7=df_hwasung_7.drop(columns=['ETL_YMD'])
df_hwasung_8=df_hwasung_8.drop(columns=['ETL_YMD'])
df_hwasung_9=df_hwasung_9.drop(columns=['ETL_YMD'])
df_hwasung_10=df_hwasung_10.drop(columns=['ETL_YMD'])
df_hwasung_11=df_hwasung_11.drop(columns=['ETL_YMD'])
df_hwasung_12=df_hwasung_12.drop(columns=['ETL_YMD'])


def make_hwasung_su(df, month):
    """
    df: 해당 월 수원시 유동인구 원본 DataFrame (예: df_suwon_7)
    month: 몇 월인지 표시용 (예: 7, 8, 9 ...)
    """
    # 필터링
    df_tmp = df[(df['ADMI_NM'] == '봉담읍') & (df['FORN_GB'] == 'N')]

    # 숫자 합계
    col_sum = df_tmp.sum(numeric_only=True)

    # 합계 행 추가
    label = f'col_sum_su_{month}'
    df_tmp.loc[label] = col_sum

    # 합계 행만 추출
    df_tmp = df_tmp.loc[[label]]

    # 필요 없는 컬럼 드랍
    df_tmp = df_tmp.drop(columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])

    return df_tmp

df_hwasung_su_11=make_hwasung_su(df_hwasung_11,11)
df_hwasung_su_12=make_hwasung_su(df_hwasung_12,12)
df_hwasung_su_1=make_hwasung_su(df_hwasung_1,1)
df_hwasung_su_2=make_hwasung_su(df_hwasung_2,2)

##수원대 7월
df_hwasung_su_7 = df_hwasung_7[df_hwasung_7['ADMI_NM'] == '봉담읍']
df_hwasung_su_7 = df_hwasung_su_7[df_hwasung_7['FORN_GB'] == '내국인']
df_hwasung_su_7

col_sum_su_7 = df_hwasung_su_7.sum(numeric_only=True)
df_hwasung_su_7.loc['col_sum_su_7'] = col_sum_su_7

df_hwasung_su_7 = df_hwasung_su_7.loc[['col_sum_su_7']]
df_hwasung_su_7 = df_hwasung_su_7.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_hwasung_su_7

##수원대 8월
df_hwasung_su_8 = df_hwasung_8[df_hwasung_8['ADMI_NM'] == '봉담읍']
df_hwasung_su_8 = df_hwasung_su_8[df_hwasung_8['FORN_GB'] == '내국인']
df_hwasung_su_8

col_sum_su_8 = df_hwasung_su_8.sum(numeric_only=True)
df_hwasung_su_8.loc['col_sum_su_8'] = col_sum_su_8

df_hwasung_su_8 = df_hwasung_su_8.loc[['col_sum_su_8']]
df_hwasung_su_8 = df_hwasung_su_8.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_hwasung_su_8

##수원대 9월
df_hwasung_su_9 = df_hwasung_9[df_hwasung_9['ADMI_NM'] == '봉담읍']
df_hwasung_su_9 = df_hwasung_su_9[df_hwasung_9['FORN_GB'] == '내국인']
df_hwasung_su_9

col_sum_su_9 = df_hwasung_su_9.sum(numeric_only=True)
df_hwasung_su_9.loc['col_sum_su_9'] = col_sum_su_9

df_hwasung_su_9 = df_hwasung_su_9.loc[['col_sum_su_9']]
df_hwasung_su_9 = df_hwasung_su_9.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_hwasung_su_9

##수원대 10월
df_hwasung_su_10 = df_hwasung_10[df_hwasung_10['ADMI_NM'] == '봉담읍']
df_hwasung_su_10 = df_hwasung_su_10[df_hwasung_10['FORN_GB'] == '내국인']
df_hwasung_su_10

col_sum_su_10 = df_hwasung_su_10.sum(numeric_only=True)
df_hwasung_su_10.loc['col_sum_su_10'] = col_sum_su_10

df_hwasung_su_10 = df_hwasung_su_10.loc[['col_sum_su_10']]
df_hwasung_su_10 = df_hwasung_su_10.drop(
    columns=['ADMI_CD','CTY_NM','ADMI_NM','TIME_CD','FORN_GB'])
df_hwasung_su_10

df_su_1_12 = pd.concat(
    [df_hwasung_su_1,
     df_hwasung_su_2,
     df_hwasung_su_7,
     df_hwasung_su_8,
     df_hwasung_su_9,
     df_hwasung_su_10,
     df_hwasung_su_11,
     df_hwasung_su_12],
    axis=0   # 행 기준 합치기
)
df_su_1_12

#경희대
df_kyunghee_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/경희대_유동인구.csv')
#아주대
df_ajou_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/아주대_유동인구.csv')
#경기대
df_gyeonggi_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/경기대_유동인구.csv')
#성대
df_skku_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/성균관대대_유동인구.csv')
#단국대
df_dku_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/단국대_유동인구.csv')
#명지대
df_myeongji_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/명지대_유동인구.csv')
#용인대
df_yu_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/용인대_유동인구.csv')
#수원대
df_su_1_12.to_csv('C:/portfolio/khuda_toyproject/데이터셋/수원대_유동인구.csv')

df_kyunghee_1_12