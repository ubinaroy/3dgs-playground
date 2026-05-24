# 3DGS Playground — Learning Workspace

本项目基于官方 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)，包含两套教学 notebook：一套针对预处理好的 COLMAP 数据集（Truck），一套是**自包含的视频到 3D 重建 pipeline**（手机视频 → ffmpeg → COLMAP → 3DGS）。三步分解数据加载流程，不调用 `Scene()` 构造函数。

## 内容总览

| 文件 | 说明 |
|------|------|
| `3dgs_walkthrough.ipynb` | 预处理数据集 notebook：从 COLMAP sparse/0/ 加载，三步分解 + 30k 训练 + 评估 + 高斯分析 |
| `video_pipeline.ipynb` | **自包含视频 pipeline**：只需改一行视频路径，自动完成 ffmpeg 抽帧 → COLMAP → 3DGS 训练 → 导出 |
| `3dgs_code_guide.html` | 3DGS 源码导读（中文） |
| `3dgs_pipeline.html` | 3DGS 全流程图文教程（中文） |
| `3dgs_pipeline.md` | 同上，Markdown 版 |

## Getting Started

```bash
# 克隆（含 submodules）
git clone --recursive https://github.com/ubinaroy/gaussian-splatting.git
cd gaussian-splatting

# 环境安装（需 CUDA 11+）
conda env create --file environment_new.yml
conda activate gaussian_splatting

# 编译 CUDA 扩展（必需！notebook 依赖这些）
pip install submodules\diff-gaussian-rasterization
pip install submodules\simple-knn
pip install submodules\fused-ssim
```

> 如果 clone 时忘了 `--recursive`，可补跑 `git submodule update --init --recursive`。

## Video Pipeline 快速开始

`video_pipeline.ipynb` 只需改 **一行代码**，其余全自动。

### Workflow

```
手机视频 .mp4                     在线查看 .ply
      │                               ▲
      ▼                               │
ffmpeg 抽帧 (-r 3)              superspl.at
      ▼                               ▲
COLMAP SfM (CPU)               下载到本地
  feature_extractor                ▲
  → sequential_matcher             │
  → mapper                  gaussians.save_ply
  → image_undistorter             ▲
      ▼                            │
   3DGS 训练 30k iter ─────────────┘
```

### 使用方式

```python
# 唯一需要修改的地方
VIDEO_PATH = "/root/autodl-tmp/您的视频.mp4"
```

运行后自动生成层级目录：

```
/root/autodl-tmp/video_data/
  └── 您的视频/              ← DATA_ROOT（自动创建）
      ├── input/             ← ffmpeg 抽帧
      ├── images/            ← 去畸变后（image_undistorter 生成）
      ├── sparse/0/          ← COLMAP SfM 结果
      ├── database.db        ← COLMAP 特征数据库
      └── output/            ← 3DGS 训练输出
          ├── final_point_cloud.ply      ← 最终 PLY（拖进 superspl.at）
          ├── final_model.pth            ← 模型 checkpoint
          ├── point_cloud/iteration_30000/point_cloud.ply
          ├── training_log.pkl
          └── iter_*.png                ← 训练过程快照
```

### kid.mp4 实测结果

| 指标 | 数值 |
|------|------|
| 视频时长 | ~30s 手机视频 |
| 抽帧 | 89 帧（3fps） |
| COLMAP 注册 | 20 帧（SIMPLE_PINHOLE） |
| 训练 PSNR | 36.86 |
| 测试 PSNR (mean) | 24.57 |
| 最终高斯数 | 344,683 |
| 训练时间 | ~15min（RTX 4080 SUPER） |

### 在线查看 PLY

无需本地安装任何工具：

1. 训练完成后下载 PLY：`scp -P [port] root@[website]:[/root/autodl-tmp/video_data/kid/output/final_point_cloud.ply] [PC/Dir]`
2. 浏览器打开 **[superspl.at](https://superspl.at)** / **[antimatter15.com/splat](https://antimatter15.com/splat)**，拖入 `.ply`

## 数据集

### 方式一：下载预处理好的 COLMAP 数据包

Truck 场景的 COLMAP 结果（含 images + sparse/）包含在官方数据包中：

```bash
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip -d data/
```

目录结构应为：
```
data/tandt/truck/
├── images/          # 251 张照片
└── sparse/          # COLMAP SfM 结果
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

然后使用 `3dgs_walkthrough.ipynb`，将 `data_path` 指向 `data/tandt/truck/`。

### 方式二：上传自己的视频，跑 video_pipeline.ipynb

1. SCP 上传视频到服务器：`scp -P [port] [video.mp4] root@[website]:/root/autodl-tmp/`
2. 用 VS Code / Jupyter 打开服务器上的 `video_pipeline.ipynb`
3. `VIDEO_PATH` 为 `/root/autodl-tmp/video.mp4`
4. 按序执行所有 cell

## 设计理念

Notebook 不调用 `Scene()` 构造函数，而是分别调用三个步骤：

1. **`readColmapSceneInfo(path)`** → 加载 COLMAP SfM 结果，得到 `SceneInfo`
2. **`cameraList_from_camInfos(...)`** → 构建相机列表 `Camera[]`
3. **`gaussians.create_from_pcd(...)`** → 从点云初始化高斯参数

每一步都可以独立检查中间数据的 shape，便于教学和调试。

## Hard-Earned Gotchas

### COLMAP 在无头服务器上

- 必须设置 `export QT_QPA_PLATFORM=offscreen`
- **feature_extractor 和 matcher 都需要禁用 GPU**：
  - `--SiftExtraction.use_gpu 0`
  - `--SiftMatching.use_gpu 0`
- 没有这两个 flag，COLMAP 会尝试创建 OpenGL context 并 `Aborted (core dumped)`

### COLMAP matcher 选择

- `sequential_matcher`：只匹配相邻帧，适合视频序列（>50 帧），耗时 ~5min
- `exhaustive_matcher`：全配对，适合 <50 帧，CPU 上数小时
- video_pipeline.ipynb 默认 `sequential_matcher`，帧数 <50 时提示可切换

### image_undistorter 输出结构

`image_undistorter` 把去畸变后的 `cameras.bin/images.bin/points3D.bin` 放在 `sparse/` 根目录，但 3DGS 期望在 `sparse/0/` 下。video_pipeline.ipynb 的 Cell 8 已自动修复此问题（将文件从 sparse/ 复制到 sparse/0/）。

### 训练相关

- `--eval` 标志每 8 张抽 1 张作测试集，小数据集会导致测试 PSNR 严重下降
- `--white_background` 用于 NeRF Synthetic 数据集；真实照片用黑背景
- 24GB VRAM 可跑满 30k 迭代

### Notebook 调试

- CUDA 扩展需在终端编译（notebook 有说明但无法替代终端操作）
- 不继承 `Scene()` 的显存占用更低，可逐 cell 检查



### 目录结构

```
/root/gaussian-splatting/         ← 代码库
├── video_pipeline.ipynb
├── 3dgs_walkthrough.ipynb
├── environment_new.yml
└── submodules/  (CUDA 扩展)

/root/autodl-tmp/                 ← 数据和视频
├── video_data/                   ← video_pipeline 数据目录
│   ├── kid/                      ← kid 视频数据 + 输出
│   └── ...
├── truck_data/                   ← Truck 场景 COLMAP 数据
├── boat.mp4 / kid.mp4            ← 上传的视频文件
└── ...
```

### Windows 文件传输（scp）

```powershell
# 上传
scp -P [port] -pw "password" [local_file] root@[website]:/root/autodl-tmp/

# 下载
scp -P [port] -pw "password" root@[website]:/root/autodl-tmp/:/path/to/file [local_dir]
```



## 引用

```
@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
