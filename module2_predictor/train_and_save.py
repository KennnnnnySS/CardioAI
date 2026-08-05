# -*- coding: utf-8 -*-
"""
CardioAI - 模块2 Part A: 模型训练与保存（一次性脚本）
=====================================================
1. 加载 data/心血管疾病.xlsx，执行与模块1相同的数据清洗（剔除血压异常记录），
   并删除 id 与原始 age(天) 字段（age 由 age_years 取代，仅作 Pipeline 输入）。
2. 构建 Pipeline:
   - FunctionTransformer: 特征工程 age(天)->age_years(四舍五入), BMI
   - ColumnTransformer: StandardScaler(连续特征) + OneHotEncoder(分类特征)
   - XGBClassifier 分类器
3. 用 joblib 将完整 Pipeline（预处理器 + 分类器）保存到 cardio_predictor_model.pkl，
   Flask API 直接加载该文件即可对新样本做与训练完全一致的预处理。

运行: python module2_predictor/train_and_save.py
"""

import os

import joblib
import numpy as np
import pandas as pd
from features import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    RAW_FEATURES,
    engineer_features,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "心血管疾病.xlsx")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardio_predictor_model.pkl")


def load_and_clean(path: str) -> pd.DataFrame:
    """与模块1相同的数据清洗，并删除 id（age 在 Pipeline 内转换为 age_years）。"""
    df = pd.read_excel(path, engine="openpyxl")
    df = df[df["ap_lo"] < df["ap_hi"]]  # 舒张压必须小于收缩压
    df = df[df["ap_hi"].between(90, 250)]  # 收缩压 [90, 250]
    df = df[df["ap_lo"].between(60, 150)]  # 舒张压 [60, 150]
    return df.drop(columns=["id"])


def main() -> None:
    df = load_and_clean(DATA_PATH)

    X = df[RAW_FEATURES]
    y = df["cardio"]

    # 特征工程步骤（age -> age_years、BMI）
    feature_engineer = FunctionTransformer(engineer_features, validate=False)

    # 预处理器: 连续特征标准化 + 分类特征独热编码
    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), CONTINUOUS_FEATURES),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # 原始 age(天) 等未选中的列直接丢弃
    )

    # 完整 Pipeline: 特征工程 -> 预处理 -> XGBoost 分类
    pipeline = Pipeline(
        steps=[
            ("engineer", feature_engineer),
            ("preprocess", preprocessor),
            ("classifier", XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1)),
        ]
    )

    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"模型已保存: {MODEL_PATH}")
    print(f"训练样本数: {len(X)}, 正例(患病)占比: {y.mean():.2%}")

    # 自检: 用前 5 条训练样本做一次预测
    proba = pipeline.predict_proba(X.head(5))[:, 1]
    pred = pipeline.predict(X.head(5))
    print("自检 - 预测患病概率:", np.round(proba, 4).tolist())
    print("自检 - 预测标签     :", pred.tolist())
    print("自检 - 真实标签     :", y.head(5).tolist())


if __name__ == "__main__":
    main()
