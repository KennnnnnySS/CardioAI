# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CardioAI - 心血管疾病智能辅助系统

Streamlit + Flask 混合架构的医学数据分析 / 预测 / 语音问答应用。数据源为 `data/心血管疾病.xlsx`（UCI 心血管疾病数据集），字段全部为数值编码：`gender` 1=女/2=男，`cholesterol`/`gluc` 1=正常/2=偏高/3=严重偏高，`cardio` 0=无疾病/1=有疾病。

## 开发流程（Git 提交规则）

**每次新增功能或修复 bug 后，必须立即执行一次 git 提交**，不要攒着一批改动才提交：

```bash
git add -A
git commit -m "<type>: <改动描述>"
```

提交信息沿用项目现有的中文风格与类型前缀：`feat:`（新功能）、`fix:`（bug 修复），例如 `feat: 新增心率异常检测模块`、`fix: 修复 BMI 计算除零报错`。改动完成后先自行运行验证再提交。

## 运行环境

- Python 虚拟环境 `cardioenv/`（已 gitignore）。激活：`source cardioenv/Scripts/activate`
- 依赖：`pip install -r requirements.txt`
- API Key 在根目录 `.env`（已 gitignore），代码用 `python-dotenv` 自动加载。必填键：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`DASHSCOPE_API_KEY`

## 启动方式

| 模块 | 命令 | 访问地址 |
|------|------|----------|
| 模块1 可视化看板 (Streamlit) | `streamlit run module1_dashboard/cardio_dashboard.py` | Streamlit 默认端口 |
| 模块2 训练模型（一次性脚本） | `python module2_predictor/train_and_save.py` | - |
| 模块2 预测 API (Flask) | `python module2_predictor/app.py` | http://127.0.0.1:5000/ |
| 模块3 语音助手 (Flask) | `python module3_voice_assistant/voice_assistant_app.py` | http://127.0.0.1:5001/ |

## 架构

三个相互独立的模块，共用同一份 Excel 数据和同一套清洗 / 特征工程逻辑：

1. **模块1 `module1_dashboard/`**：Streamlit + Plotly 数据看板。数据清洗（剔除 `ap_lo >= ap_hi` 的记录，收缩压限定 [90, 250]、舒张压 [60, 150]）与特征工程（`age` 天 → `age_years` 年、BMI）封装在 `@st.cache_data` 函数中。

2. **模块2 `module2_predictor/`**：XGBoost 预测 + Flask 服务。`train_and_save.py` 把「特征工程 → ColumnTransformer(StandardScaler + OneHotEncoder) → XGBClassifier」封装为完整 Pipeline，用 joblib 存成 `cardio_predictor_model.pkl`（可由脚本重新生成，已 gitignore）；`app.py` 提供 `POST /predict_cardio`，接收 11 个原始特征，返回 `{prediction, probability}`。

3. **模块3 `module3_voice_assistant/`**：Flask 语音问答。`/ask` 接口先用 DeepSeek（经 `langchain-openai`）生成文字回答，再用阿里云百炼 CosyVoice（`dashscope`）合成 MP3，base64 编码后返回 `{text_answer, audio_base64}`。

## 关键注意事项

- **特征定义是模块2两端的数据契约**：`module2_predictor/features.py` 中的 `RAW_FEATURES`（11 列）、`CONTINUOUS_FEATURES`、`CATEGORICAL_FEATURES`、`engineer_features()` 供训练与推理共用。改动特征必须先改这里。
- **`engineer_features` 必须可导入**：它被封装进 Pipeline 的 `FunctionTransformer` 并随 pkl 持久化，必须定义在模块文件而非 `__main__` 中，否则 joblib 反序列化时抛 AttributeError。
- **数据清洗规则需两处同步**：模块1 和模块2 各自实现了同一套血压异常剔除逻辑，改动数据集清洗规则时要同时改 `module1_dashboard/cardio_dashboard.py` 的 `clean_and_engineer()` 与 `module2_predictor/train_and_save.py` 的 `load_and_clean()`。
- **CosyVoice 实例不可复用**：`SpeechSynthesizer.call()` 结束后 WebSocket 关闭但内部状态不重置，同一实例第二次 `call()` 会抛错——每次请求必须新建实例（`create_synthesizer()` 已处理，勿"优化"掉）。
- **TTS 长度限制**：CosyVoice 单次合成有长度上限，`TTS_MAX_CHARS = 500` 仅截断用于合成，接口仍返回完整 `text_answer`。
- **路径推导**：各模块用 `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` 从自身位置向上推导 `PROJECT_ROOT` 定位 `data/` 与 `.env`，因此从任意目录启动都能工作，不要改成相对 cwd。
- **视觉设计令牌**：三个模块共用冷调临床配色——主色 `#0E7C86`（心电青绿）、标题文本 `#10212B`、次要文本 `#5C6F7A`、背景 `#F6FAFB`，见 `.streamlit/config.toml`。
