# 3D Gaussian Splatting 完整 Pipeline 记录

> 场景: Tanks & Temples **Truck**
> 日期: 2026-05-20
> 服务器: `connect.westb.seetacloud.com:18544`
> 数据源: 251 张真实卡车照片

---

## 一、环境准备

```bash
# 连接服务器
ssh -p 18544 root@connect.westb.seetacloud.com
# 密码: G3rrm2cU1a+6

# 激活 conda 环境
conda activate gaussian_splatting

# 如果没装 conda 环境:
# cd gaussian-splatting && conda env create -f environment.yml

# 安装 COLMAP（需要 CUDA 版本）
apt-get install -y colmap
```

---

## 二、数据获取

**来源**: [3DGS 官方仓库](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) 的 tandt_db.zip（682MB）

```bash
# 下载
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -O tandt_db.zip

# 提取 Truck 场景的原始照片（只提取 images，丢弃预计算的 sparse/）
mkdir -p truck_data/input
unzip -q tandt_db.zip 'tandt/truck/images/*'
mv tandt/truck/images/* truck_data/input/
rm -rf tandt
```

**最终目录结构**:
```
truck_data/
└── input/                 ← 251 张 JPG 原始照片
    ├── 000001.jpg         ← 979×546 像素
    ├── 000002.jpg
    └── ...
```

---

## 三、COLMAP SfM（从照片生成点云 + 相机位姿）

### 3.1 特征提取

```bash
export QT_QPA_PLATFORM=offscreen
mkdir -p truck_data/distorted/sparse

colmap feature_extractor \
    --database_path truck_data/distorted/database.db \
    --image_path truck_data/input \
    --ImageReader.single_camera 1 \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu 0
```

- 每张图提取 3000-7000 个 SIFT 特征
- 耗时 ~1.6 分钟

### 3.2 顺序特征匹配（重要优化）

> 251 张图如果使用穷举匹配（exhaustive_matcher），需要匹配 31,375 对，CPU 模式需数小时。
> 因为这些图是视频帧序列，改用 **sequential_matcher**，只匹配相邻帧。

```bash
colmap sequential_matcher \
    --database_path truck_data/distorted/database.db \
    --SiftMatching.use_gpu 0
```

- 耗时 ~4.6 分钟（vs 穷举的数小时）

### 3.3 稀疏重建（Mapper / Bundle Adjustment）

```bash
colmap mapper \
    --database_path truck_data/distorted/database.db \
    --image_path truck_data/input \
    --output_path truck_data/distorted/sparse \
    --Mapper.ba_global_function_tolerance=0.000001
```

- 注册 251 张图像
- 生成 **136,029 个 3D 点云**
- 最终重投影误差: 0.34 px
- 耗时 ~5 分钟

### 3.4 畸变校正

```bash
colmap image_undistorter \
    --image_path truck_data/input \
    --input_path truck_data/distorted/sparse/0 \
    --output_path truck_data \
    --output_type COLMAP

# 修正目录结构（3DGS 代码要求 sparse/0/）
mkdir -p truck_data/sparse/0
mv truck_data/sparse/*.bin truck_data/sparse/0/
```

**最终数据集结构**:
```
truck_data/
├── input/                 ← 原始照片
├── images/                ← 校正后的照片（251 张）
├── distorted/             ← COLMAP 临时文件
└── sparse/0/
    ├── cameras.bin        ← 相机内参
    ├── images.bin         ← 每张图的位姿
    └── points3D.bin       ← SfM 点云（136,029 个点）
```

---

## 四、3DGS 训练

```bash
cd gaussian-splatting
python train.py -s /path/to/truck_data -m output/truck --disable_viewer
```

### 关键参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `-s` | `truck_data` | 数据集路径 |
| `-m` | `output/truck` | 输出路径 |
| `--disable_viewer` | - | 禁用 GUI，适合无头服务器 |
| `--eval` | 不传 | 不设 eval，全部图片用于训练 |

### 训练过程

1. 初始化 **136,029** 个 3D 高斯（从 SfM 点云）
2. 每步随机挑 1 张训练图
3. 可微渲染 → 与 GT 计算 L1 + SSIM 损失
4. 反向传播更新高斯参数（位置/颜色/缩放/旋转/不透明度）
5. 迭代 500-15000: 自适应密度控制（克隆/分裂/剪枝）
6. 每 3000 步: 重置不透明度
7. 每 1000 步: 提升 SH 阶数（0→1→2→3）

### 训练结果

| 迭代 | L1 损失 | PSNR |
|---|---|---|
| 7,000 | 0.033 | 24.88 |
| 30,000 | 0.022 | **27.98** |

- SSIM: **0.9167**
- LPIPS: **0.1137**
- 训练时间: ~18 分钟
- 输出: `output/truck/point_cloud/iteration_30000/point_cloud.ply`（491MB）

---

## 五、渲染

```bash
python render.py -m output/truck --skip_test
```

- 遍历 251 个训练视角
- 使用完整 tile-based CUDA rasterizer（含 SH 全阶颜色）
- 输出到 `train/ours_30000/renders/*.png` + `gt/*.png`
- 耗时: ~2 分钟

---

## 六、Web 实时查看器

### 6.1 转换格式（.ply → .splat）

```bash
# antimatter15/splat 的 convert.py
cd /path/to/splat
python convert.py model.ply -o model.splat
```

- .ply 491MB → .splat 64MB（只保留 DC 颜色，丢掉高阶 SH）

### 6.2 启动 HTTP 服务

```bash
cd /path/to/gsplat-viewer/
python -m http.server 8080
```

> gsplat-viewer/ 目录包含:
> - `index.html` — 入口页面
> - `index.js` — 加载 gsplat.js 库 + truck.splat
> - `truck.splat` — 转换后的 3DGS 模型
> - `style.css` — 样式

### 6.3 本地访问

```powershell
# 新开终端窗口，建立 SSH 端口转发
ssh -L 8080:localhost:8080 -p 18544 root@connect.westb.seetacloud.com
# （保持窗口开着）

# 浏览器打开
http://localhost:8080
```

---

## 七、完整执行命令一览

```bash
# ===== 1. 数据 =====
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
mkdir -p truck_data/input
unzip -q tandt_db.zip 'tandt/truck/images/*'
mv tandt/truck/images/* truck_data/input/

# ===== 2. COLMAP SfM =====
export QT_QPA_PLATFORM=offscreen
mkdir -p truck_data/distorted/sparse

colmap feature_extractor --database_path truck_data/distorted/database.db --image_path truck_data/input --ImageReader.single_camera 1 --ImageReader.camera_model OPENCV --SiftExtraction.use_gpu 0

colmap sequential_matcher --database_path truck_data/distorted/database.db --SiftMatching.use_gpu 0

colmap mapper --database_path truck_data/distorted/database.db --image_path truck_data/input --output_path truck_data/distorted/sparse --Mapper.ba_global_function_tolerance=0.000001

colmap image_undistorter --image_path truck_data/input --input_path truck_data/distorted/sparse/0 --output_path truck_data --output_type COLMAP
mkdir -p truck_data/sparse/0 && mv truck_data/sparse/*.bin truck_data/sparse/0/

# ===== 3. 训练 =====
cd gaussian-splatting
python train.py -s /path/to/truck_data -m output/truck --disable_viewer

# ===== 4. 渲染 =====
python render.py -m output/truck

# ===== 5. 指标 =====
# 手动跑前 N 张（metrics.py 需 test/ 目录结构）
python -c "..."
```

---

## 八、当前服务器文件位置

| 内容 | 路径 |
|---|---|
| 原始照片 | `/root/autodl-tmp/truck_data/input/` |
| COLMAP 结果 | `/root/autodl-tmp/truck_data/sparse/0/` |
| 训练模型 | `/root/gaussian-splatting/output/truck/` |
| 渲染 PNG | `/root/gaussian-splatting/output/truck/train/ours_30000/` |
| Web 查看器 | `/root/autodl-tmp/gsplat-viewer/` |
| Web 模型文件 | `/root/autodl-tmp/gsplat-viewer/truck.splat` |

## 九、本地文件

| 文件 | 说明 |
|---|---|
| `D:\CCBench\truck_render.png` | 3DGS 渲染的卡车 |
| `D:\CCBench\truck_gt.png` | 对应的真值照片 |
| `D:\CCBench\3dgs_pipeline.md` | 本文档 |
| `D:\CCBench\3dgs_pipeline.html` | HTML 版文档 |
