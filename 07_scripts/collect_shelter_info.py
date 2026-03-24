"""
동물보호센터 정보 수집 스크립트
=============================
공공데이터포털 API를 통해 전국 동물보호센터의 운영 현황을 수집한다.

수집 항목:
- 센터명, 센터유형(직영/위탁), 주소(시도/시군구)
- 수의사 인원수, 사양관리사 인원수
- 운영시간, 휴무일, 지정일자

사용법:
    python collect_shelter_info.py

출력:
    02_outputs/data/shelter_center_info.csv
"""

import requests
import json
import csv
import time
import os
from pathlib import Path

# === 설정 ===
API_KEY = "b90a9b724c2a9eaf290d5eb0e3a1ac47e1ca4beff4019ba2f70868b9817553ec"
BASE_URL = "https://apis.data.go.kr/1543061/animalShelterSrvc_v2/shelterInfo_v2"
OUTPUT_DIR = Path(__file__).parent.parent / "02_outputs" / "data"
OUTPUT_FILE = OUTPUT_DIR / "shelter_center_info.csv"

# 출력 디렉토리 생성
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_shelter_info(page_no=1, num_of_rows=100):
    """동물보호센터 정보 API 1페이지 호출"""
    # 공공데이터포털 인증키는 URL에 직접 삽입해야 이중 인코딩 문제를 피할 수 있음
    # requests의 params에 넣으면 자동 URL 인코딩이 되어 403 발생 가능
    url = (
        f"{BASE_URL}"
        f"?serviceKey={API_KEY}"
        f"&numOfRows={num_of_rows}"
        f"&pageNo={page_no}"
        f"&_type=json"
    )

    resp = requests.get(url, timeout=30)

    # 403 등 에러 시 상세 정보 출력
    if resp.status_code != 200:
        print(f"[오류] HTTP {resp.status_code}")
        print(f"  URL: {url[:120]}...")
        print(f"  응답: {resp.text[:500]}")
        resp.raise_for_status()

    # XML 응답이 올 수 있으므로 확인
    content_type = resp.headers.get("Content-Type", "")
    if "xml" in content_type and "<resultCode>00</resultCode>" not in resp.text:
        print(f"[경고] XML 응답 수신:")
        print(f"  {resp.text[:500]}")
        # XML에서 에러 메시지 추출 시도
        if "<resultMsg>" in resp.text:
            import re
            msg = re.search(r"<resultMsg>(.*?)</resultMsg>", resp.text)
            if msg:
                print(f"  에러 메시지: {msg.group(1)}")
        return [], 0

    data = resp.json()

    # 응답 구조 파싱
    header = data.get("response", {}).get("header", {})
    result_code = header.get("resultCode", "")

    if result_code != "00":
        print(f"[오류] resultCode={result_code}, resultMsg={header.get('resultMsg', '')}")
        return [], 0

    body = data.get("response", {}).get("body", {})
    total_count = body.get("totalCount", 0)
    items = body.get("items", {})

    if not items:
        return [], total_count

    # items가 dict 안에 item 리스트로 들어있는 경우 처리
    if isinstance(items, dict):
        item_list = items.get("item", [])
    elif isinstance(items, list):
        item_list = items
    else:
        item_list = []

    # 단건인 경우 리스트로 변환
    if isinstance(item_list, dict):
        item_list = [item_list]

    return item_list, total_count


def collect_all_shelters():
    """전체 동물보호센터 정보 수집"""
    print("=" * 60)
    print("동물보호센터 정보 수집 시작")
    print("=" * 60)

    # 1페이지 호출하여 전체 건수 확인
    items, total_count = fetch_shelter_info(page_no=1, num_of_rows=100)
    print(f"\n전체 동물보호센터 수: {total_count}건")

    if total_count == 0:
        print("[경고] 데이터가 없습니다. API 키나 URL을 확인하세요.")
        # 응답 원문 출력
        params = {
            "serviceKey": API_KEY,
            "numOfRows": 5,
            "pageNo": 1,
            "_type": "json"
        }
        resp = requests.get(BASE_URL, params=params, timeout=30)
        print(f"응답 원문:\n{resp.text[:1000]}")
        return []

    all_items = list(items)

    # 나머지 페이지 수집
    total_pages = (total_count // 100) + (1 if total_count % 100 > 0 else 0)

    for page in range(2, total_pages + 1):
        print(f"  페이지 {page}/{total_pages} 수집 중...")
        time.sleep(0.5)  # API 부하 방지

        try:
            items, _ = fetch_shelter_info(page_no=page, num_of_rows=100)
            all_items.extend(items)
        except Exception as e:
            print(f"  [오류] 페이지 {page} 수집 실패: {e}")
            continue

    print(f"\n수집 완료: {len(all_items)}건")
    return all_items


def parse_shelter_data(items):
    """API 응답을 정리하여 딕셔너리 리스트로 변환"""
    parsed = []

    for item in items:
        row = {
            "센터명": item.get("careNm", ""),
            "센터유형": item.get("divisionNm", ""),  # 직영/위탁 구분
            "시도": item.get("orgdownNm", item.get("orgNm", "")),
            "시군구": item.get("jurisdNm", ""),
            "도로명주소": item.get("careAddr", ""),
            "위도": item.get("lat", ""),
            "경도": item.get("lng", ""),
            "수의사인원수": item.get("vetPersonCnt", ""),
            "사양관리사인원수": item.get("specsPersonCnt", ""),
            "전화번호": item.get("careTel", ""),
            "지정일자": item.get("dsignationDate", ""),
            "평일운영시작": item.get("weekOprStime", ""),
            "평일운영종료": item.get("weekOprEtime", ""),
            "주말운영시작": item.get("weekCellStime", ""),
            "주말운영종료": item.get("weekCellEtime", ""),
            "휴무일": item.get("closeDay", ""),
            "구조대상동물": item.get("saveTrgtAnimal", ""),
            "데이터기준일자": item.get("dataStdDt", ""),
        }
        parsed.append(row)

    return parsed


def save_to_csv(data, filepath):
    """CSV 파일로 저장"""
    if not data:
        print("[경고] 저장할 데이터가 없습니다.")
        return

    fieldnames = data[0].keys()

    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"\n저장 완료: {filepath}")
    print(f"총 {len(data)}건, 컬럼 {len(fieldnames)}개")


def print_summary(data):
    """수집 결과 요약 출력"""
    if not data:
        return

    print("\n" + "=" * 60)
    print("수집 결과 요약")
    print("=" * 60)

    # 센터유형별 집계
    type_counts = {}
    for row in data:
        t = row.get("센터유형", "미분류")
        type_counts[t] = type_counts.get(t, 0) + 1

    print("\n[센터유형별 분포]")
    for t, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {cnt}개")

    # 수의사 인원수 분포
    vet_counts = []
    for row in data:
        v = row.get("수의사인원수", "")
        if v and str(v).isdigit():
            vet_counts.append(int(v))

    if vet_counts:
        print(f"\n[수의사 인원수]")
        print(f"  수의사 정보 있는 센터: {len(vet_counts)}개")
        print(f"  평균: {sum(vet_counts)/len(vet_counts):.1f}명")
        print(f"  최소: {min(vet_counts)}명, 최대: {max(vet_counts)}명")
        print(f"  수의사 0명 센터: {vet_counts.count(0)}개")

    # 사양관리사 인원수 분포
    spec_counts = []
    for row in data:
        v = row.get("사양관리사인원수", "")
        if v and str(v).isdigit():
            spec_counts.append(int(v))

    if spec_counts:
        print(f"\n[사양관리사 인원수]")
        print(f"  정보 있는 센터: {len(spec_counts)}개")
        print(f"  평균: {sum(spec_counts)/len(spec_counts):.1f}명")
        print(f"  최소: {min(spec_counts)}명, 최대: {max(spec_counts)}명")

    # 시도별 분포
    sido_counts = {}
    for row in data:
        s = row.get("시도", "미분류")
        sido_counts[s] = sido_counts.get(s, 0) + 1

    print(f"\n[시도별 분포]")
    for s, cnt in sorted(sido_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {cnt}개")


def main():
    # 1. 전체 보호센터 정보 수집
    raw_items = collect_all_shelters()

    if not raw_items:
        print("\n[안내] 데이터 수집에 실패했습니다.")
        print("다음을 확인해주세요:")
        print("  1. API 인증키가 유효한지")
        print("  2. 인터넷 연결이 정상인지")
        print("  3. API 서비스가 활성 상태인지")
        return

    # 2. 응답 원본 저장 (디버깅용)
    raw_path = OUTPUT_DIR / "shelter_center_info_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)
    print(f"\n원본 JSON 저장: {raw_path}")

    # 3. 데이터 파싱 및 정리
    parsed_data = parse_shelter_data(raw_items)

    # 4. CSV 저장
    save_to_csv(parsed_data, OUTPUT_FILE)

    # 5. 요약 출력
    print_summary(parsed_data)

    print("\n" + "=" * 60)
    print("다음 단계:")
    print("  1. shelter_center_info.csv를 확인")
    print("  2. agg_shelter_performance.csv와 매칭")
    print("  3. 직영/위탁별 입양률 분석")
    print("=" * 60)


if __name__ == "__main__":
    main()
