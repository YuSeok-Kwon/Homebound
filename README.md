# Homebound

보호소 동물이 "집으로 향하는" 여정을 데이터로 추적하는 프로젝트.

## 프로젝트 개요
- **생성일**: 2026-03-23
- **분석 목적**: 보호종료동물 데이터(79만건)를 활용하여 입양률 향상 및 보호소 운영 효율화 전략 도출
- **데이터 출처**: 보호종료동물 상세 데이터

## 디렉토리 구조
```
Homebound/
├── 01_raw_data/              # 원본 데이터 (읽기 전용)
│   ├── images/               #   보호동물 이미지 (63장)
│   └── 보호종료동물 상세 데이터.csv
├── 02_outputs/               # 산출물 통합
│   ├── data/                 #   가공된 데이터 (집계, Tableau용 등)
│   └── figures/              #   시각화 결과
├── 03_notebooks/             # Jupyter 노트북 (분석별 서브폴더)
├── 04_docs/                  # 문서 (참고자료, 리포트)
├── 05_src/                   # 재사용 가능한 Python 모듈
│   ├── data/                 #   데이터 로드/전처리
│   ├── features/             #   피처 엔지니어링
│   ├── visualization/        #   시각화 함수
│   └── utils.py              #   유틸리티
├── 06_analysis/              # 하위 분석 모듈
├── 06_models/                # 학습된 모델 파일
├── 07_scripts/               # CLI 실행용 스크립트
├── 999_Temporary/            # 임시 파일 (주기적 정리)
├── config/                   # 설정 파일
├── CLAUDE.md
├── README.md
└── requirements.txt
```

## 환경 설정
```bash
pip install -r requirements.txt
```

## 파일 네이밍 규칙
- 형식: `YYYYMMDD_Homebound_분석내용_버전.확장자`
- 예시: `20260323_Homebound_EDA_v1.0.ipynb`
- `final`, `최종` 사용 금지 → 날짜 + 버전(`v1`, `v2`) 사용
