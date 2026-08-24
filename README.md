# 智能座舱出行规划 Agent

## 一键启动与页面测试

Windows 下在项目根目录执行：

```powershell
.\start-all.cmd
```

也可以直接双击根目录的 `start-all.cmd`。脚本会自动检查依赖、启动 FastAPI 后端和 Vite 前端，等待服务就绪后打开：

- 页面：[http://127.0.0.1:5173](http://127.0.0.1:5173)
- 后端健康检查：[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- 后端接口文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

页面测试文本：

```text
我想去杭州玩两天，预算3000元，两个人，想去西湖和博物馆，喜欢杭帮菜，希望住交通方便的酒店，整体轻松一点。
```

启动窗口中按 `Ctrl+C` 会同时关闭前后端。运行日志保存在 `.dev-runtime/`，该目录不会进入 Git。

## 后端模块

后端入口：

- `backend/app/main.py`

启动方式：

```powershell
python -m uvicorn backend.app.main:app --reload
```

接口：

- `GET /`
- `GET /health`
- `POST /api/asr/transcribe`
- `POST /api/extract/keywords`
- `POST /api/adapter/candidate-pool`
- `POST /api/pipeline/gather-candidates`

调试地址：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

ASR 模块目录：

- `backend/app/api/asr.py`：语音识别接口
- `backend/app/api/extract.py`：关键词提取接口
- `backend/app/api/adapter.py`：候选池适配接口
- `backend/app/services/asr/`：语音识别服务层
- `backend/app/services/extract/`：关键词提取服务层
- `backend/app/services/adapter/`：候选池适配服务层
- `backend/app/models/asr.py`：语音识别返回模型
- `backend/app/models/extract.py`：关键词提取返回模型
- `backend/app/models/adapter.py`：候选池适配返回模型
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

关键词提取结果中包含：

- 基础提取字段：目的地、日期、预算、人数、景点/餐饮/酒店偏好、出行风格
- `algorithm_input`：给排序模型输入端使用的结构化对象

`algorithm_input` 当前包含：

- `search_context`
- `objective_weights`
- `subjective_preference`
- `fusion_config`
- `sequence_model_input`

候选池适配使用方式：

1. 先准备好关键词提取结果和 `place / food / hotel` 的 `detail.json`
2. 启动后端服务
3. 打开 `/docs`
4. 调用 `POST /api/adapter/candidate-pool`
5. 可选传入：
   - `extract_result_path`
   - `place_detail_path`
   - `food_detail_path`
   - `hotel_detail_path`
6. 如果不传三类 detail 路径，接口会默认读取三个模块下最新一次 output 的 `detail.json`
7. 如果提取结果里的目的地无法直接解析坐标，适配层会优先回退到 `food / hotel output` 中的查询中心作为候选池中心点

候选池适配输出位置：

- `data/candidate_pool/`

候选池适配结果中包含：

- 更新后的 `algorithm_input`
- `spot_candidates`
- `food_candidates`
- `hotel_candidates`
- `meta`

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

```powershell
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

```powershell
python scripts/getdata/food/main.py
```

输出位置：

- `scripts/getdata/food/output/`

### hotel 模块

入口：

- `scripts/getdata/hotel/main.py`

用途：

- 抓取或生成酒店候选数据，并统一输出为候选池可读取的结构

运行方式：

```powershell
python scripts/getdata/hotel/main.py
```

交互输入只有目的地、搜索半径、结果数量、入住日期和入住晚数；输出目录由程序自动命名为 `hotel_candidates_入住日期_时间戳`。

数据口径：酒店名称、地址、坐标和 POI ID 来自高德；距离由程序计算；评分、评论数、房价、房型、库存和设施为 MVP 原型模拟字段。

输出位置：

- `scripts/getdata/hotel/output/`

## 数据目录

- `data/asr_text/`：语音识别后的纯文本
- `data/extract_result/`：关键词提取后的结构化 JSON
- `data/candidate_pool/`：统一候选池结果
- `data/mock/`：后续 mock 数据

## 文档目录

- `doc/流程图版开发进度.md`：当前进度文档

## 测试联调脚本

入口：
- `tests/run_random_pipeline_demo.py`

用途：
- 随机生成一段旅游需求文本
- 自动调用关键词提取模块
- 自动运行 `place / food / hotel` 三个采集脚本
- 自动调用候选池适配层和 ranking，输出一份带排序字段的候选列表

说明：

- 这是联调演示脚本，不是独立的自动化测试。
- 脚本会访问高德等外部服务，并写入本地 output 和 data 目录。
- 外部服务异常时，完整联调可能失败；建议先使用固定 mock 数据验证接口契约。

运行方式：
```powershell
python tests/run_random_pipeline_demo.py
```

可选参数：
```powershell
python tests/run_random_pipeline_demo.py --seed 7
```

输出位置：
- 联调汇总：`tests/output/`
- 关键词提取结果：`data/extract_result/`
- 候选池结果：`data/candidate_pool/`

## 自动化测试

安装测试依赖并运行固定 mock 测试：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

当前共有 10 个测试，覆盖酒店高德请求与 fallback、酒店交互输入、adapter 契约、ranking Top-K/偏好/近期 POI 抑制，以及 pipeline 中文城市映射和酒店五项输入契约。自动化测试不访问真实外网。

## 依赖安装

```powershell
python -m pip install -r requirements.txt
```

开发与测试依赖：

```powershell
python -m pip install -r requirements-dev.txt
```

高德 Key 按当前双人协作约定保存在地点、餐饮和酒店脚本的代码常量中。酒店脚本在请求失败或结果为空时会使用本地坐标与模拟酒店回退。

## 当前 MVP 状态

当前主链路为：

```text
文本输入 -> extract -> 数据采集 -> adapter -> ranking -> frontend
```

已实现：

- `extract`：解析用户需求并生成 `algorithm_input`
- `adapter`：统一 `place / food / hotel` 数据为候选池
- `ranking`：对评分、距离、热度、偏好、预算及近期访问历史进行可解释融合排序，稳定输出 Top-K、`final_score`、`rank` 和 `score_breakdown`
- `pipeline`：已实际调用 ranking，并向前端返回排序候选
- `frontend`：已实现文本输入、处理状态、候选结果、三条展示路线和详情弹窗

当前边界：

- ranking 当前是三类候选的分组排序，不是完整的路线组合优化。
- 前端三条路线按固定模式拼装，属于演示展示逻辑，不代表已经完成时间、空间和行程顺序规划。
- 音频文件转写接口已存在，但实时语音输入和真实音频回归仍未完成。
- 已建立 10 个固定 mock 测试，并验证一键启动时后端健康接口和前端首页可用；真实页面联调仍依赖外部服务。
- 酒店真实杭州 POI 抓取已验证成功，但价格、评分、库存等业务字段仍是原型模拟数据。

当前优先事项：

1. 使用一键启动脚本完成一次真实页面文本输入到结果展示的 E2E，并记录耗时和失败日志。
2. 将固定 mock 测试接入 CI。
3. 明确标记页面中的模拟价格、评分和库存字段。
4. 继续明确“候选排序”和“路线组合展示”的验收边界。
