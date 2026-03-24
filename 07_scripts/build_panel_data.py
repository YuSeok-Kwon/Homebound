"""
시군구별 연도별 사회경제 패널 데이터 구축 스크립트
- 인구/세대 (시군구 수준, 2019~2024)
- 재정자립도 (시군구 수준, 2019~2024)
- 면적 (시군구 수준)
- 연령별 인구 → 고령화율 (시도 수준, 시군구에 시도값 매핑)
- 세대원수별 세대 → 1인가구비율 (시도 수준, 시군구에 시도값 매핑)
"""
import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings('ignore')

BASE = "/sessions/busy-intelligent-euler/mnt/02_Homebound"
EXT = f"{BASE}/01_raw_data/external"
OUT = f"{BASE}/02_outputs/data"
os.makedirs(OUT, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023, 2024]

# ============================================================
# 1. 인구/세대 데이터 (시군구 수준) — 16개 시도별 파일 + 서울 파일
# ============================================================
print("=" * 60)
print("1. 시군구별 인구/세대 데이터 통합")
print("=" * 60)

pop_files = [f for f in os.listdir(EXT) if f.startswith("201912_202512_주민등록인구및세대현황")]
pop_dfs = []

for f in sorted(pop_files):
    df = pd.read_csv(os.path.join(EXT, f), encoding='cp949')
    # 행정구역에서 코드 추출
    df['행정구역_원본'] = df['행정구역'].astype(str)
    df['행정구역명'] = df['행정구역'].str.extract(r'^(.+?)\s*\(')[0].str.strip()
    df['행정구역코드'] = df['행정구역'].str.extract(r'\((\d+)\)')[0]
    pop_dfs.append(df)
    print(f"  {f}: {len(df)}행")

pop_all = pd.concat(pop_dfs, ignore_index=True)
print(f"  -> 통합: {len(pop_all)}행")

# 시도/시군구 구분 (코드 끝 8자리가 00000000이면 시도)
pop_all['is_sido'] = pop_all['행정구역코드'].str[-8:] == '00000000'

# 연도별 컬럼 추출: 총인구수, 세대수
panel_pop = []
for year in YEARS:
    cols_map = {}
    for col in pop_all.columns:
        if f'{year}년_총인구수' in col:
            cols_map['총인구수'] = col
        elif f'{year}년_세대수' in col:
            cols_map['세대수'] = col
        elif f'{year}년_남자 인구수' in col or f'{year}년_남자인구수' in col:
            cols_map['남자인구수'] = col
        elif f'{year}년_여자 인구수' in col or f'{year}년_여자인구수' in col:
            cols_map['여자인구수'] = col
    
    if '총인구수' not in cols_map:
        print(f"  [WARN] {year}년 총인구수 컬럼 없음")
        continue
    
    subset = pop_all[['행정구역명', '행정구역코드', 'is_sido']].copy()
    subset['연도'] = year
    
    for key, col in cols_map.items():
        subset[key] = pd.to_numeric(pop_all[col].astype(str).str.replace(',', ''), errors='coerce')
    
    panel_pop.append(subset)

df_pop = pd.concat(panel_pop, ignore_index=True)
# 시군구만 (시도 제외)
df_pop_sigungu = df_pop[~df_pop['is_sido']].copy()
print(f"  시군구 인구 패널: {len(df_pop_sigungu)}행 (시군구 x 연도)")
print(f"  시군구 수: {df_pop_sigungu.groupby('연도')['행정구역명'].nunique().to_dict()}")

# ============================================================
# 2. 재정자립도 (시군구 수준, 2019~2024)
# ============================================================
print("\n" + "=" * 60)
print("2. 재정자립도 통합")
print("=" * 60)

fiscal_dfs = []
for year in YEARS:
    f = f"재정자립도[최종] {year}.csv"
    fp = os.path.join(EXT, f)
    if not os.path.exists(fp):
        print(f"  [WARN] {f} 없음")
        continue
    df = pd.read_csv(fp, encoding='utf-8-sig')
    print(f"  {f}: {len(df)}행, 컬럼={list(df.columns)}")
    fiscal_dfs.append(df)

df_fiscal = pd.concat(fiscal_dfs, ignore_index=True)
print(f"  재정자립도 통합: {len(df_fiscal)}행")
print(f"  컬럼: {list(df_fiscal.columns)}")
print(f"  연도별 행수: {df_fiscal.groupby('회계연도').size().to_dict()}")

# ============================================================
# 3. 면적 (시군구 수준)
# ============================================================
print("\n" + "=" * 60)
print("3. 행정구역별 면적")
print("=" * 60)

df_area = pd.read_csv(os.path.join(EXT, "행정구역별_면적_및_축적_20260323180622.csv"), 
                       encoding='utf-8-sig', header=[0,1])
print(f"  면적: {len(df_area)}행")
print(f"  컬럼(상위): {[c[0] for c in df_area.columns[:7]]}")
print(f"  컬럼(하위): {[c[1] for c in df_area.columns[:7]]}")
print(f"  첫 5행:")
print(df_area.head())

# ============================================================
# 4. 연령별 인구 → 고령화율 (시도 수준)
# ============================================================
print("\n" + "=" * 60)
print("4. 연령별 인구 → 고령화율 (시도)")
print("=" * 60)

df_age = pd.read_csv(os.path.join(EXT, "201912_202412_연령별인구현황_연간.csv"), encoding='cp949')
print(f"  행: {len(df_age)}, 컬럼: {len(df_age.columns)}")

df_age['행정구역명'] = df_age['행정구역'].str.extract(r'^(.+?)\s*\(')[0].str.strip()

aging_panel = []
for year in YEARS:
    total_col = f'{year}년_계_총인구수'
    cols_65plus = [c for c in df_age.columns if f'{year}년_계_' in c and 
                   any(age in c for age in ['60~69세', '70~79세', '80~89세', '90~99세', '100세'])]
    
    if total_col not in df_age.columns:
        continue
    
    total = pd.to_numeric(df_age[total_col].astype(str).str.replace(',',''), errors='coerce')
    
    # 65세 이상 = 70~79 + 80~89 + 90~99 + 100이상 + 60~69의 일부
    # 정확한 65세 이상 계산을 위해 70+ 만 사용 (보수적 추정) 또는 60대 포함
    elderly_cols = [c for c in df_age.columns if f'{year}년_계_' in c and 
                    any(age in c for age in ['70~79세', '80~89세', '90~99세', '100세 이상'])]
    col_60 = [c for c in df_age.columns if f'{year}년_계_60~69세' in c]
    
    elderly = sum(pd.to_numeric(df_age[c].astype(str).str.replace(',',''), errors='coerce') for c in elderly_cols)
    # 60대의 약 50%가 65세 이상으로 추정 (보수적)
    if col_60:
        sixty = pd.to_numeric(df_age[col_60[0]].astype(str).str.replace(',',''), errors='coerce')
        elderly = elderly + sixty * 0.5  # 60~64: 약 50%, 65~69: 약 50%
    
    aging_rate = (elderly / total * 100).round(2)
    
    temp = pd.DataFrame({
        '시도명': df_age['행정구역명'],
        '연도': year,
        '고령화율_시도': aging_rate
    })
    aging_panel.append(temp)

df_aging = pd.concat(aging_panel, ignore_index=True)
# 전국 행 제거
df_aging = df_aging[df_aging['시도명'] != '전국'].copy()
print(f"  고령화율 패널: {len(df_aging)}행")
print(f"  예시: {df_aging.head(3).to_string()}")

# ============================================================
# 5. 세대원수별 세대 → 1인가구비율 (시도 수준)
# ============================================================
print("\n" + "=" * 60)
print("5. 세대원수별 세대 → 1인가구비율 (시도)")
print("=" * 60)

df_hh = pd.read_csv(os.path.join(EXT, "201912_202512_주민등록인구기타현황(세대원수별 세대수)_year.csv"), 
                     encoding='cp949')
print(f"  행: {len(df_hh)}, 컬럼: {len(df_hh.columns)}")

df_hh['행정구역명'] = df_hh['행정구역'].str.extract(r'^(.+?)\s*\(')[0].str.strip()

single_panel = []
for year in YEARS:
    total_col = f'{year}년12월_전체세대'
    single_col = f'{year}년12월_1인세대'
    
    if total_col not in df_hh.columns:
        print(f"  [WARN] {year}년 세대 데이터 없음")
        continue
    
    total = pd.to_numeric(df_hh[total_col].astype(str).str.replace(',',''), errors='coerce')
    single = pd.to_numeric(df_hh[single_col].astype(str).str.replace(',',''), errors='coerce')
    
    temp = pd.DataFrame({
        '시도명': df_hh['행정구역명'],
        '연도': year,
        '1인가구비율_시도': (single / total * 100).round(2)
    })
    single_panel.append(temp)

df_single = pd.concat(single_panel, ignore_index=True)
df_single = df_single[df_single['시도명'] != '전국'].copy()
print(f"  1인가구비율 패널: {len(df_single)}행")
print(f"  예시: {df_single.head(3).to_string()}")

# ============================================================
# 6. 통합
# ============================================================
print("\n" + "=" * 60)
print("6. 최종 통합")
print("=" * 60)

# 시군구에 시도 매핑을 위해 시도명 추출
def extract_sido(name):
    """시군구명에서 시도 추출"""
    sido_list = ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시',
                 '대전광역시', '울산광역시', '세종특별자치시', '경기도', '강원도', '강원특별자치도',
                 '충청북도', '충청남도', '전라북도', '전북특별자치도', '전라남도', 
                 '경상북도', '경상남도', '제주특별자치도']
    for sido in sido_list:
        if name and sido in str(name):
            return sido
    return None

# 인구 데이터에 시도명 추가
# 시도 행에서 시도명 매핑 만들기
sido_map = df_pop[df_pop['is_sido']][['행정구역명', '행정구역코드']].drop_duplicates()
# 시군구 코드의 앞 2자리가 시도 코드
df_pop_sigungu['시도코드'] = df_pop_sigungu['행정구역코드'].str[:2]

# 시도명 매핑
sido_code_name = {}
for _, row in sido_map.iterrows():
    code2 = str(row['행정구역코드'])[:2]
    sido_code_name[code2] = row['행정구역명']

df_pop_sigungu['시도명'] = df_pop_sigungu['시도코드'].map(sido_code_name)

print(f"  시도명 매핑 결과:")
print(f"  {df_pop_sigungu['시도명'].value_counts().head(10).to_string()}")

# 최종 패널: 인구 + 시도별 고령화율/1인가구비율 조인
panel = df_pop_sigungu[['행정구역명', '행정구역코드', '시도명', '연도', '총인구수', '세대수']].copy()

# 고령화율 조인
panel = panel.merge(df_aging, on=['시도명', '연도'], how='left')
# 1인가구비율 조인
panel = panel.merge(df_single, on=['시도명', '연도'], how='left')

print(f"\n  통합 패널 (면적/재정자립도 조인 전): {len(panel)}행")
print(f"  결측 현황:")
print(f"  {panel.isnull().sum().to_string()}")

# 저장
panel.to_csv(f"{OUT}/socioeconomic_panel_v1.csv", index=False, encoding='utf-8-sig')
print(f"\n  저장: {OUT}/socioeconomic_panel_v1.csv")
print(f"  최종 행수: {len(panel)}")
print(f"  연도별 시군구 수: {panel.groupby('연도')['행정구역명'].nunique().to_dict()}")
print(f"\n  샘플:")
print(panel.head(10).to_string())

