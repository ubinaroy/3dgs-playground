# 3DGS Playground — Learning Workspace

本项目基于官方 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)，围绕 COLMAP 真实照片（Truck 场景，251 张）搭建了一套教学级的学习环境。不调用 `Scene()` 构造函数，而是三步分解数据加载流程，适合深入理解 3DGS 的 pipeline。

## 内容总览

| 文件 | 说明 |
|------|------|
| `3dgs_walkthrough.ipynb` | 主 notebook：三步分解 (readColmapSceneInfo → cameraList_from_camInfos → create_from_pcd)，完整的 30k 训练、渲染、评估、高斯分析流程 |
| `3dgs_code_guide.html` | 3DGS 源码导读（中文） |
| `3dgs_pipeline.html` | 3DGS 全流程图文教程（中文） |
| `3dgs_pipeline.md` | 同上，Markdown 版 |

## Getting Started

```bash
# 克隆（含 submodules）
git clone --recursive https://github.com/ubinaroy/gaussian-splatting.git
cd gaussian-splatting

# 环境安装（需 CUDA 11+）
conda env create --file environment_new.yml # 而不是原始 yml，看你环境
conda activate gaussian_splatting

# 编译 CUDA 扩展（必需！notebook 依赖这些）
pip install submodules\diff-gaussian-rasterization
pip install submodules\simple-knn
pip install submodules\fused-ssim
```

然后打开 `3dgs_walkthrough.ipynb` 按序执行即可。

> 如果 clone 时忘了 `--recursive`，可补跑 `git submodule update --init --recursive`。

## 数据集

Notebook 使用 Tanks & Temples 的 **Truck** 场景。有两种获取方式：

### 方式一：下载预处理好的 COLMAP 数据包（推荐）

Truck 场景的 COLMAP 结果（含 images + sparse/）包含在官方数据包中：

```bash
# 下载 Tanks & Temples 数据集（约 650MB）
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

然后 notebook 中的 `data_path` 指向 `data/tandt/truck/` 即可。

### 方式二：用你自己的照片

参考官方 `convert.py` 流程：
```bash
# 1. input 目录放你的照片
mkdir -p my_scene/input && cp your_photos/* my_scene/input/

# 2. 运行 COLMAP（需安装 COLMAP）
python convert.py -s my_scene

# 3. 完成后 my_scene/ 下会生成 images/ + sparse/0/
```

## 设计理念

Notebook 不调用 `Scene()` 构造函数，而是分别调用三个步骤：

1. **`readColmapSceneInfo(path)`** → 加载 COLMAP SfM 结果，得到 `SceneInfo`
2. **`cameraList_from_camInfos(...)`** → 构建相机列表 `Camera[]`
3. **`gaussians.create_from_pcd(...)`** → 从点云初始化高斯参数

每一步都可以独立检查中间数据的 shape，便于教学和调试。

## Hard-Earned Gotchas

### 训练相关
- `--eval` 标志每 8 张抽 1 张作测试集，小数据集会导致测试 PSNR 严重下降（~13），仅当需要评估时才用
- `--white_background` 用于 NeRF Synthetic 数据集；真实照片用黑背景
- 24GB VRAM 可跑满 30k 迭代；非 `--eval` 模式全部 251 图参与训练

### COLMAP 相关
- `sequential_matcher`（匹配相邻帧）适合 >50 张的视频帧序列；`exhaustive_matcher`（全配对）仅 <50 张时可用
- 251 张图片 exhaustive 匹配在 CPU 上需要数小时，sequential 仅 ~5 分钟
- **Headless 服务器必须** `export QT_QPA_PLATFORM=offscreen` + `--SiftExtraction.use_gpu 0`
- `convert.py --skip_matching` 会跳过提取+匹配+mapper，如果 `sparse/0/` 不存在，后续 `image_undistorter` 会失败

### Notebook 调试
- CUDA 扩展需在终端编译（notebook cell 0.1 有说明但无法替代终端操作）
- 不继承 `Scene()` 的显存占用更低，可逐 cell 检查；但需要手动传递 `pipe` 参数（`compute_cov3D_python=False`）

## 服务器说明

完整的训练代码、编译好的 CUDA 扩展、训练输出和数据集部署在服务器上：

```
服务器: connect.westc.seetacloud.com
用户: root
工作目录: /root/gaussian-splatting/
数据集: /root/autodl-tmp/truck_data/ (COLMAP 已完成)
```

本仓库包含完整的 submodules（`SIBR_viewers/`、`submodules/diff-gaussian-rasterization`、`submodules/simple-knn`、`submodules/fused-ssim`），克隆时需 `--recursive`。以下内容不纳入 git 追踪（文件可在部署服务器上获取）：

- `output*/` — 训练输出（ply/pth 文件可能很大）
- `*.ply`, `*.pth`, `*.splat` — 模型权重和点云文件

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
