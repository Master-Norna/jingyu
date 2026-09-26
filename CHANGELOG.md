# 更新记录

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
