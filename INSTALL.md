# UBODT Manager 安装指南

## 快速安装

### 方法1: 使用安装脚本（推荐）

```bash
# 1. 编译项目
cd build
cmake ..
make ubodt_daemon -j4

# 2. 返回项目根目录并运行安装脚本
cd ..
./install_tools.sh
```

安装脚本会自动检测环境并安装到合适的位置：
- 如果在conda环境中：安装到 `$CONDA_PREFIX/bin`
- 否则：安装到 `$HOME/.local/bin`

### 方法2: 手动安装到 conda 环境

```bash
# 编译
cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make ubodt_daemon -j4

# 复制到 conda bin 目录
cp ubodt_daemon $CONDA_PREFIX/bin/
chmod +x $CONDA_PREFIX/bin/ubodt_daemon
```

### 方法3: 使用 make install（需要 root 权限）

```bash
cd build
cmake ..
make ubodt_daemon
sudo make install
```

这会安装到 `/usr/local/bin`（需要 sudo 权限）。

### 方法4: 手动安装到自定义位置

```bash
# 创建本地 bin 目录（如果不存在）
mkdir -p $HOME/.local/bin

# 复制可执行文件
cp build/ubodt_daemon $HOME/.local/bin/
chmod +x $HOME/.local/bin/ubodt_daemon

# 添加到 PATH（如果还没有）
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 验证安装

安装完成后，在任意目录运行：

```bash
ubodt_daemon --help
```

如果显示帮助信息，说明安装成功！

## 使用示例

```bash
# 启动 UBODT daemon（在后台持续运行）
ubodt_daemon start --ubodt data/ubodt.bin

# 查看daemon状态
ubodt_daemon status

# 在任何终端都可以查询状态
ubodt_daemon status

# 运行 FMM
fmm --config config.xml

# 停止 daemon
ubodt_daemon stop
```

## 其他工具

安装脚本也会同时安装以下工具：
- `fmm` - 快速地图匹配
- `cmm` - 协方差地图匹配
- `stmatch` - 空间时间匹配
- `h3mm` - H3地图匹配
- `ubodt_gen` - UBODT生成器
- `ubodt_converter` - UBODT格式转换器
- `ubodt_daemon` - **UBODT守护进程（后台持久化存储）**

## 故障排除

### 命令未找到

如果运行 `ubodt_daemon` 时提示 "command not found"：

1. 检查安装位置：
```bash
which ubodt_daemon
```

2. 如果没有输出，检查 PATH：
```bash
echo $PATH
```

3. 如果安装位置不在 PATH 中，添加到 PATH：
```bash
export PATH="/path/to/install/dir:$PATH"
```

### 权限问题

如果遇到权限错误：

```bash
# 使用 sudo
sudo make install

# 或者安装到用户目录（推荐）
./install_tools.sh
```

### 重新安装

如果需要重新安装：

```bash
# 删除旧版本
rm $CONDA_PREFIX/bin/ubodt_daemon

# 重新编译和安装
cd build
make ubodt_daemon
cp ubodt_daemon $CONDA_PREFIX/bin/
```

## 卸载

```bash
# 从 conda 环境卸载
rm $CONDA_PREFIX/bin/ubodt_daemon

# 或从系统目录卸载
sudo rm /usr/local/bin/ubodt_daemon

# 或从用户目录卸载
rm $HOME/.local/bin/ubodt_daemon
```

## 更多信息

详细的使用说明请参考：
- `UBODT_MANAGER_README.md` - UBODT Manager 使用指南
- 项目 README.md - 整体项目文档
