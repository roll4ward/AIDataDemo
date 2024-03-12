import concurrent
import streamlit as st
import pandas as pd
import numpy as np
import functools

import matplotlib.pyplot as plt

from SmartFarmDataMartAPI.src.api import SmartFarmAPI
from SmartFarmDataMartAPI.codes.appendix import FatrCode

from utils import timeit



# Global instance
api = SmartFarmAPI("config.yaml")


def define_page():
    st.set_page_config(page_title="이상치 탐지 데모", page_icon="📈")
    
    st.markdown("# 이상치 탐지 데모")
    st.sidebar.header("이상치 탐지 데모")  
    st.write(
        """연도, 지역, 품종에 기반하여 실제 농장들에서 수집된 데이터를 기반으로 "내부CO2" 값의 정상치를 시각화합니다. 
        버튼을 누르면 그래프를 그릴 수 있습니다. 반응형 그래프로 그리고 이상치인지 아닌지를 판별하는 기능을 추가해야 합니다.
        """
    )
    
    
    st.title("이상치 탐지 프로토타입")
    loading_state = st.text("") # 로딩
    year = st.multiselect("연도", [year for year in range(2015, 2025)], default=[2020, 2021])

    st.subheader('농장 정보')

    try:
        loading_state.text("데이터 로딩중입니다.")
        data = getCroppingSeasonDataList(year)
        data = pd.DataFrame(data)
        loading_state.text("")

        regions = st.multiselect("지역", data["addressName"].unique(), default=[data.iloc[0]["addressName"]])
        data = data[data["addressName"].isin(regions)]

        crop_items = st.multiselect("재배 작물", data["itemName"].unique(), default=[data.iloc[0]["itemName"]])
        data = data[data["itemName"].isin(crop_items)]
        st.write(data)
    except:
        st.write("데이터가 없습니다.")

    st.subheader('시각화 정보')
    
    cropSrlNums = data["croppingSerlNo"].astype(int).to_list()

    progress_bar = st.progress(0)
    progress_status = st.empty()
    
    if st.button("시각화 하기 (1분 이상 소요)"):
        data_famrs = getCroppingSeasonEnvDataList(cropSrlNums)
        st.pyplot(draw_linegraph_by_category_monthly(data_famrs, "내부CO2"))





@st.cache_data()
def getCroppingSeasonDataList(years):
    data = []
    for year in years:
        data += api.crop_season.getCroppingSeasonDataList(year, verbose=False)
    
    return data


@timeit
@functools.cache
def fetch_env_data_for_crop(cropSrlNum):
    num_pages = int(float(api.crop_season.getCroppingSeasonEnvDataList(cropSrlNum, 1, verbose=False)[0]["totalPage"]))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list of futures for the API calls
        futures = [executor.submit(api.crop_season.getCroppingSeasonEnvDataList, cropSrlNum, page, verbose=False) for page in range(1, num_pages + 1)]
        
        # Wait for the futures to complete and gather the results
        env_datas = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Concatenate the data frames
    df_env_data = pd.concat([pd.DataFrame(df) for df in env_datas])
    
    return df_env_data

def getCroppingSeasonEnvDataList(cropSrlNums):
    # Use ThreadPoolExecutor to parallelize fetching across different cropSrlNums
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_env_data_for_crop, cropSrlNum) for cropSrlNum in cropSrlNums]
        df_env_datas = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Concatenate all the data frames into one
    data = pd.concat(df_env_datas)
    
    data = data[["measDate", "fldCode", "sectCode", "fatrCode", "senVal"]]
    data["senVal"] = data["senVal"].astype("double")
    data["measDate"] = pd.to_datetime(data["measDate"])
    
    df_pivoted = data.iloc[:].pivot_table(index='measDate', columns='fatrCode', values='senVal', aggfunc="mean")
    df_pivoted.rename(columns=lambda x: FatrCode.get_description(x), inplace=True)
    
    return df_pivoted


def draw_linegraph_by_category_monthly(data, category):

    # Assuming 'data' is your DataFrame and it has a DateTimeIndex
    # Group the DataFrame by month
    monthly_groups = data.groupby(pd.Grouper(freq='M'))

    # Calculate global minimum and maximum values for 내부CO2(ppm)
    global_min = data[category].min()
    global_max = data[category].max()

    # Create a 6x6 grid of subplots
    fig, axs = plt.subplots(3, 2, figsize=(30, 20))  # Adjusted figsize for readability
    fig.suptitle(f'Monthly {category} Data Plots')

    # Flatten the array of axes for easy iteration
    axs_flat = axs.flatten()

    # Iterate over each group (month) and plot its data
    for ax, (name, group) in zip(axs_flat, monthly_groups):
        # Scatter plot for 내부CO2(ppm) against hour of the day
        hours = group.index.hour
        
        co2 = group[category]
        ax.scatter(hours, co2, alpha=0.5, label=category)
        
        # Prepare data for the box plot
        box_data = [group[group.index.hour == hour][category].dropna() for hour in range(24)]
        
        # Add box plot over the scatter plot
        # Find positions for the box plots. In this case, we're aligning them with hours, but slightly offset to avoid overlap
        positions = np.arange(24)
        ax.boxplot(box_data, positions=positions, widths=0.4, manage_ticks=False, patch_artist=True)
        
        # Set the y-axis limits based on the global min and max values
        ax.set_ylim(global_min, global_max)
        
        ax.set_title(name.strftime('%Y-%m'))
        ax.set_xticks(np.arange(0, 24, 1))  # Ensure x-ticks represent each hour
        ax.set_xticklabels(np.arange(0, 24, 1))  # Rotate x-axis labels for clarity
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel(category)
        ax.grid(True)

    # Hide any unused subplots
    for ax in axs_flat[monthly_groups.ngroups:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
    return fig





if __name__ == "__main__":
    define_page()
