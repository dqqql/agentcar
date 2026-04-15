# 智能座舱出行规划 Agent


详细进度见：

- [doc/流程图版开发进度.md](doc/流程图版开发进度.md)


## 目录结构

```text
backend/
  app/
    api/
    models/
    services/
    utils/
data/
  mock/
doc/
frontend/
scripts/
  getdata/
    food/
    hotel/
    place/
    output_utils.py
tests/
```

## 目录说明

- `backend/`：后端代码
- `frontend/`：前端演示页
- `data/`：业务数据与 mock 数据
- `scripts/getdata/`：客观数据抓取与原型数据生成脚本
- `doc/`：项目文档与进度记录
- `tests/`：测试代码

## 数据脚本说明

`scripts/getdata` 下目前有三类脚本：

- `place/`：地点 / 景点类数据抓取
- `food/`：餐饮类数据抓取
- `hotel/`：酒店候选数据生成

三个脚本现在都使用统一的输出目录结构，各自产物都写到本目录下的 `output/` 中。仓库默认忽略 `output/`、`__pycache__/` 和历史导出数据，保证其他人拉取代码时只拿到脚本本身。

## 本地运行

在项目根目录执行：

```bash
uvicorn backend.app.main:app --reload
```

启动后访问：

- [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)




