import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# 设置页面配置
st.set_page_config(
    page_title="Agentic Labs Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 创建示例数据
@st.cache_data
def create_example_data():
    data = {
        'age': ['[90-100]', '[70-80]', '[60-70]', '[50-60]'],
        'gender': ['Male', 'Female', 'Male', 'Female'],
        'glipizide': ['No', 'No', 'No', 'No'],
        'admission_type_id': [1, 1, 1, 2],
        'num_medications': [15, 14, 5, 45],
        'number_diagnoses': [9, 5, 4, 6],
        'discharge_disposition_id': [6, 3, 1, 18],
        'num_procedures': [0, 0, 0, 6],
        'readmitted': ['>30', 'NO', '>30', 'NO']
    }
    return pd.DataFrame(data)

# 主页面
st.markdown("### Use our example or upload your own dataset")

# 文件上传区域
uploaded_file = st.file_uploader(
    "To replace the example dataset, drag & drop a CSV file here, or click to select a file",
    type="csv"
)

# 显示说明文本
st.write("""
This page already loads an example dataset by default. You can also upload your own dataset if you want to override the default one. 
Upload any dataset for which you want to predict one column. We support most tabular datasets without specific formatting. The 
supported dataset size in this demo is currently limited to 100,000 total cells (rows × columns) for training and test datasets combined.
""")

st.write("Our example dataset is a simplified version of the [Diabetes130US dataset](https://example.com). The goal is to predict if / when a diabetic patient will be readmitted to the hospital.")

# 加载数据
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = create_example_data()

# 数据预览
st.subheader("Dataset Preview")
st.dataframe(df)

# 创建两列布局
col1, col2 = st.columns(2)

# 目标列选择
with col1:
    st.write("Target Column")
    target_column = st.selectbox("", df.columns.tolist(), index=df.columns.get_loc('readmitted'))

# 任务类型选择
with col2:
    st.write("Task")
    task_type = st.selectbox("", ["Classification"], index=0)

# 特征列选择
st.write("Feature Columns")
feature_columns = st.multiselect(
    "Select feature columns",
    df.columns.tolist(),
    default=[col for col in df.columns if col != 'readmitted']
)

# 创建两列布局用于性能评估和测试集预测
col1, col2 = st.columns(2)

# 性能评估部分
with col1:
    st.subheader("Performance Estimation")
    st.write("Estimate the performance of our TabPFN2 model on your dataset compared to classic baselines like XGBoost and Random Forest.")
    if st.button("Estimate Performance"):
        # 这里添加模型性能评估的模拟函数
        st.info("Performance estimation in progress...")

# 测试集预测部分
with col2:
    st.subheader("Test Set Prediction")
    st.write("Get predictions from our model on a specific test set.")
    test_file = st.file_uploader(
        "To replace the example dataset, drag & drop a CSV file here, or click to select a file",
        type="csv",
        key="test_file"
    )
    
    # 创建示例测试数据
    test_data = pd.DataFrame({
        'age': ['[80-90]', '[70-80]', '[40-50]', '[50-60]'],
        'gender': ['Male', 'Male', 'Male', 'Female'],
        'glipizide': ['No', 'No', 'No', 'Steady'],
        'admission_type_id': [5, 2, 1, 1],
        'num_medications': [10, 7, 10, 22],
        'number_diagnoses': [8, 8, 9, 7],
        'discharge_disposition_id': [11, 1, 1, 1]
    })
    
    st.dataframe(test_data)
    
    if st.button("Predict"):
        # 这里添加预测的模拟函数
        st.info("Prediction in progress...")

