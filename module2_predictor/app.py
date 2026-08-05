# -*- coding: utf-8 -*-
"""
CardioAI - 模块2 Part B: Flask API 服务
========================================
提供 /predict_cardio 接口：接收 11 个原始特征的 JSON POST 请求，
加载 cardio_predictor_model.pkl 中的完整 Pipeline 进行预处理与预测，
返回患病概率 probability 与预测结果 prediction (0/1)。

运行: python module2_predictor/app.py
访问: http://127.0.0.1:5000/  （前端表单） 或直接 POST /predict_cardio
"""

import os

import joblib
import pandas as pd
from features import RAW_FEATURES
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cardio_predictor_model.pkl")

app = Flask(__name__)
_model = None


def get_model():
    """惰性加载模型，避免每个请求都从磁盘读取。"""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_cardio", methods=["POST"])
def predict_cardio():
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({"error": "请求体必须是 JSON 对象"}), 400

    # 校验必填字段
    missing = [f for f in RAW_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"缺少字段: {missing}"}), 400

    # 组装为单行 DataFrame（Pipeline 内部会自动完成 age->age_years、BMI 等特征工程）
    try:
        row = pd.DataFrame([[data[f] for f in RAW_FEATURES]], columns=RAW_FEATURES)
        row = row.astype(float)
    except (TypeError, ValueError) as e:
        return jsonify({"error": f"输入值不合法（需为数值）: {e}"}), 400

    model = get_model()
    try:
        probability = float(model.predict_proba(row)[0, 1])  # 患病概率
        prediction = int(model.predict(row)[0])  # 0 = 无疾病, 1 = 有疾病
    except Exception as e:  # noqa: BLE001 - 将模型异常统一转为可读的 500
        return jsonify({"error": f"预测失败: {e}"}), 500

    return jsonify(
        {
            "prediction": prediction,
            "probability": round(probability, 4),
        }
    )


if __name__ == "__main__":
    # debug=False：生产/演示模式
    app.run(host="0.0.0.0", port=5000, debug=False)
