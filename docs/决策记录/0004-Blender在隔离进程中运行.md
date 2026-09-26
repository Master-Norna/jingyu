# ADR-0004 Blender 在隔离子进程中运行

- 状态：已采纳（2026-09-26）

## 背景

bpy 庞大、依赖上下文、跨版本有变化，崩溃或卡死会带走宿主进程。MCP 服务器需要长期稳定
运行，而 Blender 可能是用户安装的可执行文件，也可能是 pip 装的 bpy 模块。

## 决定

- 每次渲染启动一个 worker 子进程，通过 `request.json` / `response.json` 文件交换
  （`jingyu.worker.v1`），主机只信任响应文件。
- 超时整组进程终止；没有响应即判定崩溃；输出全部进 `blender.log`。
- worker 只依赖纯标准库核心和 bpy；场景在主机侧已校验并补全默认值。
- bpy 做二次封装（`blender/kit.py`）：只用 `bpy.data`，统一单位与原点；版本差异集中在
  `blender/compat.py`，最低支持 4.2。

## 后果

- 每次渲染多出一次 Blender 启动的开销（秒级），换来隔离和可替换的运行时。
- 几何和材质的计算在 worker 内由同一份纯 Python 代码完成，主机侧可以单独测试。
