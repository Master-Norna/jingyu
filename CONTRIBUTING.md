# 参与贡献

景语欢迎代码、生成器、文档、测试和看图反馈。贡献应帮助模型和人获得更可控、可定位、
可复现的生图流程，而不是把审美决定藏进不可解释的默认值。

完整规范见 [docs/工程规范.md](docs/工程规范.md)，这里是摘要。

## 开发环境

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -e ".[mcp,dev]"     # Python 3.11 可加 blender 附加项安装 bpy
```

提交前运行：

```bash
ruff check . && ruff format --check .
mypy
python -m jingyu.reference --check        # 生成的文档与 Schema 是否最新
python -m pytest -q
```

没有 Blender 运行时时，`blender` 标记的测试会跳过；改动 `jingyu/blender/`、
`jingyu/bridge/` 或渲染流程时，请在有 Blender 的环境里跑 `python -m pytest -q -m blender`。
EEVEE 与 Workbench 的测试需要 `JINGYU_TEST_GPU_ENGINES=1`。

## 变更原则

- 新形状、新表面先考虑用已有生成器的参数表达；不行再加生成器，不加模型文件。
- 生成器是纯函数，每个参数写描述、默认值和范围；加完运行
  `python -m jingyu.reference --write`。
- 失败要明确：新的失败情形先在 `jingyu/errors.py` 注册错误码；不静默换成别的东西继续。
- 协议有版本、解析严格；不兼容的改动升主版本并写迁移说明。
- Blender 版本差异只写在 `jingyu/blender/compat.py`。
- 模块名、字段、id、错误码用 ASCII 英文；文档可以用中文。
- 修 bug 先写能复现它的测试。
- 不提交虚拟环境、工作区、候选、渲染输出或本机缓存。

## 第三方材料

不要提交来源不清或未经授权的模型、贴图、HDRI、图片或代码。资产补缺只接受 CC0 或许可
明确允许再分发的来源，并记录来源、版本、许可证原文和文件摘要。

项目自研贡献按 Apache-2.0 提交；提交者应有权提供相关内容。

## 提交

直接提交到 `main`，提交信息用一句话概括结果；用户可观察的变化记进 `CHANGELOG.md`。
原始项目署名见 `NOTICE`；修改版应按 Apache-2.0 醒目标明自己的修改。
