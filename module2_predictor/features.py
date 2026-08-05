# -*- coding: utf-8 -*-
"""
CardioAI - 模块2 共享特征定义与特征工程
=========================================
供 train_and_save.py（训练）与 app.py（推理）复用，保证两端字段与处理逻辑一致。

engineer_features 会被封装进 Pipeline 的 FunctionTransformer 并随模型一起保存，
因此必须定义在可导入的模块里（不能是 __main__），否则 joblib 反序列化时
无法定位该函数（AttributeError）。
"""

import numpy as np
import pandas as pd

# 11 个原始特征（已去掉 id 与标签 cardio）
RAW_FEATURES = [
    "age", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active",
]

# 连续特征（经特征工程后送入 StandardScaler）
CONTINUOUS_FEATURES = ["age_years", "bmi", "height", "weight", "ap_hi", "ap_lo"]
# 分类特征（送入 OneHotEncoder）
CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """特征工程: age(天)->age_years(四舍五入), BMI = weight / (height/100)^2。

    放在 Pipeline 内部执行，保证 API 推理时对单条新样本也能自动完成。
    """
    df = df.copy()
    df["age_years"] = np.round(df["age"] / 365.0).astype(int)
    df["bmi"] = df["weight"] / ((df["height"] / 100.0) ** 2)
    return df
