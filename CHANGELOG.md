# 更新记录

## 0.4.0（未发布）

第四波：按《景语待办汇总》落实三次实测和原定路线里还没做的事。

- CLI：`jingyu tools` 列出每个工具的参数（类型、必填或默认值、几选一），
  `jingyu tools <名字>` 打印它完整的输入输出 Schema。
- CLI：裁切图的文件名按 x0-y0-x1-y1（或 u0-v0-u1-v1）的阅读顺序命名。
- 警告降噪：编组可以写 `accept_warnings`，对组里每件物体生效；承托别的物体的东西被完全
  挡住时不再报 `frame.not_visible`。
- mypy 固定按 Linux 平台检查，本地在 Windows 上跑的结果与 CI 一致。

## 0.3.0（未发布）

第三波：看清结果。起因是第二次实测里，颜色跑偏、光落在哪、房间漏光都只能靠模型猜或
手算，而且几条警告是误报。

- 光照诊断通道：每次渲染附带 `light_mask.png` 和 `light_map.json`，用射线求出每个采样点
  被哪些灯直接照到、背光还是被谁挡住，哪些物体在取景范围内，相机周围多开敞。
- 新工具 `describe_light`：每盏灯照到各物体的比例、主要投影（灯、投影物、承影物）、
  开敞度。`locate_in_candidate` 也给出某点被哪些灯照到。
- 颜色测量：定位与布局给出物体的渲染颜色、材质颜色和偏移（色相、饱和度、明度）。
- 新视图：`light`（某盏灯照不到处压暗）、`saturation`；所有视图支持 `region` 裁切放大。
- 室内漏光检查：有窗的墙配日光而房间没封顶时报 `light.open_to_sky`。
- 警告降噪：`frame.not_visible` 只报取景范围内被挡住的物体；穿插提示只在合理时建议
  `rest_on`；新增 `attached_to`（有意相接）和 `accept_warnings`（场景级与物体级）。
- CLI：`call` 保存的图片按候选、视图和裁切区域命名，不再互相覆盖。
- 示例 `morning-window.json` 补上墙和天花板，封闭房间。
- 器皿轮廓改为曲率连续的单调曲线：原来的三次曲线在腹部、颈部控制点处曲率突变，光滑釉面上会看到高光折断。

## 0.2.0（未发布）

第二波：让模型改图更省力、看图更准。起因是第一次模型实测（清晨窗边静物）暴露的问题，
按一般问题来解决。

- 环境族：场景外面的光成为第三类生成器（`world.environment`）。`daylight` 用太阳高度、
  方位、云量、雾霾和地面颜色描述从黎明到黄昏、从晴到阴的日光，自动保持太阳方向、天空、
  阳光色温和阴影软硬一致；`uniform` 是均匀光。
- 编组与放置关系：`groups` 与 `parent` 让一组东西一起动；`rest_on` 自动把物体落到支撑
  面上（桌面、碗底），不用再手算高度。
- 新几何算子 `wall`：可以开门洞和窗洞的墙。
- 灯光：`temperature_k` 用色温代替颜色；面光加 `size_y`，可以做矩形软光。
- 物理检查：`physics.intersection`（穿插）和 `physics.floating`（悬空）警告，带修改提示。
- 画面检查：渲染后报告出画的物体（`frame.not_visible`）和曝光问题（`frame.underexposed`、
  `frame.overexposed`），写进回执。
- 新工具 `edit_scene`：用 JSON Patch 加 `merge` 改场景，`@<id>` 按 id 寻址，可以从某个
  候选的场景出发（`from_candidate`）。新工具 `diff_candidates`：两个候选之间的场景变化和
  画面布局变化。
- `list_generators` 增加环境族；场景可写 `intent`（一句话意图）作为看图时的对照。
- 看图指引重写：先写意图；先看图、用一句话说出画面，再用数字去找位置；每轮只改一个想法。
- 示例场景 `morning-window.json`。

## 0.1.0（未发布）

第一波：地基。

- 生成器框架：几何算子与材质族共用一个带参数声明的抽象，场景 Schema、默认值、工具目录
  和文档都由它推出。
- 几何算子：`box`、`plane`、`sphere`、`cylinder`、`cone`、`lathe`、`vessel`。全部输出
  闭合流形；`vessel` 用底、腹、颈、口四个半径和位置描述一件器物。
- 材质族：`ceramic`、`glass`、`metal`（九种实测反射率）、`plastic`、`principled`。
- 场景描述 `jingyu.scene.v1`：严格解析、Schema 与语义三步校验，问题带 JSON Pointer 和提示。
- Blender 二次封装与隔离渲染：worker 子进程、版本化文件协议、超时与崩溃检测；支持
  Blender 4.2 起的可执行文件和 bpy 模块；Cycles、EEVEE、Workbench。
- 不可变候选：原子发布、回执与文件哈希、完整性校验。
- 精确 ID 遮罩；按点、归一化坐标或区域定位物体；布局统计；减法审查（`hide`）。
- 看图视图：`full`、`glance`、`flip`、`squint`、`grayscale`、`values`、`id_mask`、`compare`。
- 《景语视觉宪法》v0.1 与按条款查询。
- 13 个工具，CLI 与 MCP 服务器共用；环境诊断 `jingyu doctor`。
- 工程规范、架构、场景描述与决策记录文档；错误码、工具、生成器目录与 Schema 由代码生成。
- CI：静态检查、三平台单元测试、bpy 真实渲染、wheel 隔离安装。
- 验收示例：八件器物，同一个 `vessel` 函数、八行参数。
