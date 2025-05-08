import pandas as pd
from pathlib import Path

def load_emdcd_from_region(sido: str, sigungu: str, csv_path: str = None) -> list[str]:
    """
    주어진 시/군/구 명칭에 해당하는 법정동 코드 목록을 반환합니다.

    Parameters
    ----------
    csv_path : str
        법정동 코드 CSV 파일 경로
    sido : str
        시/도 명칭 (예: '서울특별시')
    sigungu : str
        시/군/구 명칭 (예: '강남구')

    Returns
    -------
    list[str]
        해당 지역의 emdCd 목록 (중복 제거)
    """

    if csv_path is None:
        csv_path = Path(__file__).parent.parent.parent / "data" / "전국법정동.csv"

    df = pd.read_csv(csv_path, dtype=str)

    filtered = df[(df["시도명"] == sido) & (df["시군구명"] == sigungu)]

    if filtered.empty:
        raise ValueError(f"해당 지역을 찾을 수 없습니다: {sido} {sigungu}")

    return filtered["법정동코드"].dropna().unique().tolist()