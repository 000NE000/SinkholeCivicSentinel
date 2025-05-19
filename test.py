import pandas as pd

df = pd.read_csv("/Users/macforhsj/Desktop/SinkholeCivicSentinel/src/models/sinkhole_modeling/run_enhanced/enhanced_gnn_results/hotspot_analysis_full.csv")

import re

def extract_gu_dong(address):
    if pd.isna(address):
        return None, None
    # 예: "대한민국 서울특별시 동대문구 답십리로68길 24"
    gu_match = re.search(r'서울특별시\s*([\w가-힣]+구)', address)
    dong_match = re.search(r'([\w가-힣]+동)', address)

    gu = gu_match.group(1) if gu_match else None
    dong = dong_match.group(1) if dong_match else None

    return gu, dong

# 3. 새로운 컬럼으로 구, 동 저장
df[['gu', 'dong']] = df['address'].apply(
    lambda addr: pd.Series(extract_gu_dong(addr))
)

# 4. 구-동 기준으로 그룹화 후 개수 집계
summary = df.groupby(['gu', 'dong']).size().reset_index(name='count')

# 5. 정렬 (선택적으로)
summary = summary.sort_values(by=['gu', 'count'], ascending=[True, False])

# 6. 결과 확인
print(summary)