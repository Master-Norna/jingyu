# 景语 / Jingyu

**让模型用三维场景画图：可控、一致、可编辑。**

景语是一套给模型使用的生图工具与工作流。模型写一份声明式的场景描述，景语在隔离的
Blender 进程里渲染它；模型看图、指出哪里不对，景语把那个位置翻译回场景里可以修改的
字段；改完再渲染。每一张图都能追回到它的场景，每一次渲染都保留下来可以比较。

![八件器物](docs/images/八件器物.jpg)

*上图八件器物（杯、碗、花瓶、酒瓶、玻璃罐、铜罐、银瓶、玻璃杯）全部来自同一个
`vessel` 函数，每件只是一行参数：[`examples/scenes/eight-vessels.json`](examples/scenes/eight-vessels.json)。*

## 核心想法

- **生成器是三维场景加虚拟相机。** 光影、透视、遮挡由渲染器物理地保证同源；不接扩散
  模型，保持轻量。
- **收集生成器，不收集资产。** 形状来自几何算子，表面来自材质族，都是带参数的原语。
  差异再小，也只是参数值不同，不需要新文件。
- **能指出来就能改。** 每次渲染附一张精确的 ID 遮罩：指向画面上一个点或一块区域，就
  得到那里的物体、材质和它们在场景里的 JSON Pointer。
- **立宪先行。** 《景语视觉宪法》按条款查询，用来提问，保证多轮迭代不跑题。

## 快速开始

需要 Python 3.11+，以及一个 Blender 运行时（Blender 4.2 或更新的可执行文件，或者
Python 3.11 下 pip 安装的 `bpy` 模块）。

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e ".[mcp]"    # Python 3.11 且没装 Blender 时用 ".[mcp,blender]"

jingyu doctor                        # 找到了哪个 Blender、有哪些引擎和 GPU
jingyu --workspace ./workspace render examples/scenes/eight-vessels.json
```

`render` 输出新候选的 id，候选目录在 `./workspace/candidates/<id>/`。然后：

```bash
jingyu --workspace ./workspace view <候选id> squint       # 眯眼看整体
jingyu --workspace ./workspace locate <候选id> --xy 640 300   # 这里是什么
jingyu --workspace ./workspace layout <候选id>             # 画面怎么分布
jingyu --workspace ./workspace render --final examples/scenes/eight-vessels.json
```

Blender 装在非默认位置时，用环境变量 `JINGYU_BLENDER` 指向可执行文件。

## 接入模型（MCP）

```json
{
  "mcpServers": {
    "jingyu": {"command": "jingyu-mcp", "args": ["--workspace", "/path/to/workspace"]}
  }
}
```

模型先调用 `get_guide`，按其中的流程工作：`list_generators` 查原语目录，写场景，
`validate_scene`，`render_scene` 预览，`view_candidate` 看图，`locate_in_candidate`
定位，改场景再渲染。全部工具见 [docs/工具.md](docs/工具.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [技术方案](docs/技术方案.md) | 定位、工作流、同级工具与推进顺序 |
| [架构](docs/架构.md) | 分层、进程边界、候选与定位 |
| [场景描述](docs/场景描述.md) | `jingyu.scene.v1` 的格式与约定 |
| [生成器目录](docs/生成器目录.md) | 全部几何算子与材质族的参数 |
| [工具](docs/工具.md) | CLI 与 MCP 共用的工具 |
| [错误码](docs/错误码.md) | 稳定错误码 |
| [工程规范](docs/工程规范.md) | 代码、协议与数据的规范 |
| [决策记录](docs/决策记录/README.md) | 重要决定的背景与后果 |
| [景语视觉宪法 v0.1](docs/视觉宪法/景语视觉宪法-v0.1.md) | 视觉原则与追问 |

## 状态

第一波（地基）已完成：生成器框架与首批几何算子、材质族，场景描述与校验，Blender 封装
与隔离渲染，不可变候选与 ID 遮罩，看图与定位工具，CLI 与 MCP。下一波是风化层。
进度见 [CHANGELOG](CHANGELOG.md)。

## 许可

软件按 [Apache-2.0](LICENSE) 发布，署名见 [NOTICE](NOTICE)。
