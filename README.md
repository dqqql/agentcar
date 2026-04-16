# 智能座舱出行规划 Agent

## 后端模块

后端入口：

- `backend/app/main.py`

启动方式：

```bash
python -m uvicorn backend.app.main:app --reload
```

接口：

- `GET /`
- `GET /health`
- `POST /api/asr/transcribe`
- `POST /api/extract/keywords`

调试地址：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

ASR 模块目录：

- `backend/app/api/asr.py`：语音识别接口
- `backend/app/api/extract.py`：关键词提取接口
- `backend/app/services/asr/`：语音识别服务层
- `backend/app/services/extract/`：关键词提取服务层
- `backend/app/models/asr.py`：语音识别返回模型
- `backend/app/models/extract.py`：关键词提取返回模型
- `backend/app/core/config.py`：ASR 配置

ASR 使用方式：

1. 启动后端服务
2. 打开 `/docs`
3. 在 `POST /api/asr/transcribe` 上传音频文件
4. 接口返回识别文字和文本文件路径

支持的音频格式：

- `.wav`
- `.mp3`
- `.m4a`
- `.webm`
- `.ogg`
- `.flac`

ASR 输出位置：

- 临时音频：`backend/.tmp/asr/`
- 识别文本：`data/asr_text/`

关键词提取使用方式：

1. 启动后端服务
2. 打开 `/docs`
3. 在 `POST /api/extract/keywords` 里二选一：
   - 直接传 `text`
   - 传 `text_file_path`
4. 接口返回结构化关键词结果和结果文件路径

关键词提取输出位置：

- 提取结果：`data/extract_result/`

## 数据脚本模块

数据脚本目录：

- `scripts/getdata/place/`
- `scripts/getdata/food/`
- `scripts/getdata/hotel/`

统一输出规则：

- 每个脚本都写到各自目录下的 `output/`
- 每次输出一个独立任务目录
- 每个任务目录包含：
  - `summary.csv`
  - `detail.json`

### place 模块

入口：

- `scripts/getdata/place/main.py`

用途：

- 抓取地点 / 景点类数据

运行方式：

```bash
python scripts/getdata/place/main.py
```

输出位置：

- `scripts/getdata/place/output/`

### food 模块

入口：

- `scripts/getdata/food/main.py`

用途：

- 抓取餐饮类数据

运行方式：

```bash
python scripts/getdata/food/main.py
```

输出位置：

- `scripts/getdata/food/output/`

### hotel 模块

入口：

- `scripts/getdata/hotel/main.py`

用途：

- 生成酒店候选原型数据

运行方式：

```bash
python scripts/getdata/hotel/main.py
```

输出位置：

- `scripts/getdata/hotel/output/`

## 数据目录

- `data/asr_text/`：语音识别后的纯文本
- `data/extract_result/`：关键词提取后的结构化 JSON
- `data/mock/`：后续 mock 数据

## 文档目录

- `doc/流程图版开发进度.md`：当前进度文档

## 依赖安装

```bash
python -m pip install -r requirements.txt
```
