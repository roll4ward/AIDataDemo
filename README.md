# 앞구르기 👋
우리 앞구르기는 가치 있는 미래의 산업으로 스마트팜에 주목하였습니다.
스마트팜을 바닥부터 구르며 구축하고 여기서 만나는 문제를 해결하는 과정에서
성장하기 위해 만들어진 동아리입니다.

## What AI/Data team do? 
1. 데이터 수집 및 마이닝을 통해 문제를 분석하고 정의합니다.
2. AI 기술을 활용하여 기존에 알려진 open problems를 해결합니다.
3. 새로운 기능을 제시하고 이전의 문제를 해결하여 차별성을 갖추는 것을 지향합니다. 

## How to use?
```shell
cd AIDataDemo
conda create -n demo python=3.10
conda activate demo

python -m pip install requests
python -m pip install pyyaml
python -m pip install streamlit

cd SmartFarmDataMartAPI
python -m pip install -e .
cd ..
streamlit run Hello.py
```

## Demos
### 1. 📈 이상치 탐지 (anomaly detect)
> 같은 연도에, 같은 지역에 있는 그리고 같은 품종을 재배하는 농장으로부터 수집된 데이터를 기반으로
> 특정 센서값의 시간별 정상 범위를 구하여 이를 시각화합니다.
> 현재는 "내부CO2" 값을 기준으로 시각화 되었습니다.

### Contact us
- Github page   - [roll4ward](https://github.com/roll4ward)
- Discord       - [디스코드](https://discord.gg/UgxHZ5nAve)
