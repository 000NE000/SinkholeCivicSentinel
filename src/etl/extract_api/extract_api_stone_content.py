import requests
import geopandas as gpd
from shapely.geometry import shape
from src.utils.config import API_CONFIG

# VWorld가 지원하는 EPSG 코드 목록
SUPPORTED_CRS = {
    '4326': 'WGS84',
    '5179': 'Korean_2000_TM'
}

def normalize_crs(crs: str) -> str:
    code = crs.split(':')[-1]
    return code if code in SUPPORTED_CRS else '4326'

import numpy as np

# bbox 분할 함수

def split_bbox(bbox, step_deg=0.1):
    minx, miny, maxx, maxy = bbox
    xs = np.arange(minx, maxx, step_deg)
    ys = np.arange(miny, maxy, step_deg)
    cells = []
    for x0 in xs:
        for y0 in ys:
            cells.append((x0, y0, x0 + step_deg, y0 + step_deg))
    return cells

class VWorldAPIError(Exception):
    """Raised when the VWorld API returns an error status."""
    pass


def fetch_stone_content(
    api_key: str,
    bbox: tuple = None,
    emdCd: str = None,
    page_size: int = 1000,
    crs: str = 'EPSG:4326',
    subdivide: bool = True,
) -> gpd.GeoDataFrame:
    base_url = 'https://api.vworld.kr/req/data'
    crs_code = normalize_crs(crs)
    target_crs = 'EPSG:5179'

    # 공통 요청 파라미터
    request_params = {
        'service': 'data',
        'version': '2.0',
        'request': 'GetFeature',
        'key': api_key,
        'format': 'json',
        'data': 'LT_C_ASITSURSTON',
        'size': page_size,
        'geometry': 'true',
        'attribute': 'true',
        'crs': f'EPSG:{crs_code}',
    }

    # 요청 타일 설정
    if emdCd:
        tiles = [{'emdCd': emdCd}]
    elif bbox:
        if subdivide:
            tiles = [{'bbox': cell} for cell in split_bbox(bbox, step_deg=0.1)]
        else:
            tiles = [{'bbox': bbox}]
    else:
        raise ValueError('Either bbox or emdCd must be provided.')

    records = []
    for tile in tiles:
        params = request_params.copy()
        if 'emdCd' in tile:
            params['attrFilter'] = f"emdCd:={tile['emdCd']}"
        else:
            minx, miny, maxx, maxy = tile['bbox']
            params['geomFilter'] = f"BOX({minx},{miny},{maxx},{maxy})"

        page = 1
        while True:
            params['page'] = page
            resp = requests.get(base_url, params=params)
            raw = resp.json()
            data = raw.get('response', raw)

            status = data.get('status')
            if status == 'NOT_FOUND':
                break
            if status != 'OK':
                err = data.get('error', {})
                raise VWorldAPIError(
                    f"API error {err.get('code','<no code>')}: {err.get('text','<no text>')}"
                )

            features = data.get('result', {}).get('featureCollection', {}).get('features', [])
            for feat in features:
                props = feat.get('properties', {})
                geom = shape(feat.get('geometry', {}))
                records.append({
                    'emd_cd': props.get('emdCd'),
                    'stone_code': props.get('code_sg'),
                    'stone_label': props.get('label'),
                    'geometry': geom
                })

            # 페이지 정보 정수 변환해서 비교
            page_info = data.get('page', {})
            current = int(page_info.get('current', page))
            total = int(page_info.get('total', page))
            if current >= total:
                break
            page += 1

    # GeoDataFrame 생성
    gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=f'EPSG:{crs_code}')
    if crs_code != '5179':
        gdf = gdf.to_crs(target_crs)
    return gdf

if __name__ == '__main__':
    from src.utils.config import API_CONFIG
    bbox_wgs84 = (126.5, 37.2, 127.5, 37.8)
    df = fetch_stone_content(API_CONFIG['VWORLD_API_KEY'], bbox=bbox_wgs84)
    print('Rows:', len(df))
    print(df.total_bounds)