# UBODT Daemon 使用指南

## 为什么需要 Daemon？

当你使用 `ubodt_manager load` 时，UBODT 只在**当前进程**的内存中，命令执行完毕后进程退出，UBODT 也就释放了。这就是为什么你在另一个终端看不到加载的UBODT。

**解决方案：使用 `ubodt_daemon`**

Daemon（守护进程）会在后台持续运行，保持UBODT在内存中，所有其他的FMM/CMM进程都可以共享这个已加载的UBODT。

## UBODT Daemon vs UBODT Manager

| 特性 | ubodt_manager | ubodt_daemon |
|------|---------------|--------------|
| 进程生命周期 | 命令执行完就退出 | 持续在后台运行 |
| UBODT持久化 | ❌ | ✅ |
| 跨终端共享 | ❌ | ✅ |
| 多进程共享 | ❌ | ✅ |
| 用途 | 简单的一次性操作 | 批量处理、服务化 |

## 基本使用

### 1. 启动 Daemon

```bash
# 启动守护进程并加载UBODT
ubodt_daemon start --ubodt data/ubodt.bin

# 使用更大的哈希表
ubodt_daemon start --ubodt data/ubodt.bin --multiplier 2
```

**输出示例：**
```
========================================
UBODT Daemon Started
========================================
PID: 12345
UBODT: data/ubodt.bin
Rows: 1234567
Load time: 2.34s
========================================

Daemon is now running. Press Ctrl+C to stop.
```

### 2. 查看 Daemon 状态

```bash
ubodt_daemon status
```

**输出示例：**
```
========== UBODT Daemon Status ==========
Status: Running
PID: 12345
UBODT file: data/ubodt.bin
Loaded: Yes
Rows: 1234567
==========================================
```

### 3. 在另一个终端验证

打开新的终端窗口：
```bash
ubodt_daemon status
```

你会看到相同的输出！因为daemon在后台持续运行。

### 4. 停止 Daemon

```bash
ubodt_daemon stop
```

**输出示例：**
```
Stopping daemon (PID: 12345)...
Signal sent successfully.
Daemon stopped successfully.
```

## 工作流程

### 场景：批量处理多个轨迹文件

```bash
# 终端1：启动daemon
ubodt_daemon start --ubodt data/ubodt.bin

# 终端2、3、4...：并行运行FMM/CMM
fmm --config config1.xml --ubodt data/ubodt.bin
fmm --config config2.xml --ubodt data/ubodt.bin
fmm --config config3.xml --ubodt data/ubodt.bin

# 所有进程都会共享daemon中的UBODT，无需重复加载！

# 处理完成后
ubodt_daemon stop
```

## 实现原理

### 文件系统通信

Daemon使用以下文件来管理状态：

1. **PID文件**: `/tmp/ubodt_daemon.pid`
   - 存储daemon的进程ID
   - 用于发送停止信号

2. **状态文件**: `/tmp/ubodt_daemon_status.txt`
   - 存储daemon的状态信息
   - 包含：PID、UBODT文件路径、加载状态、行数等

### 信号处理

- **SIGTERM** (15): 优雅停止daemon
- **SIGINT** (2): Ctrl+C停止daemon
- **SIGKILL** (9): 强制杀死daemon（最后手段）

## 与 FMM/CMM 集成

⚠️ **重要提示**：当前版本的FMM/CMM应用仍使用进程内的UBODTManager，**不会自动使用daemon中的UBODT**。

### 当前使用方式

即使daemon在运行，FMM/CMM仍会：
1. 检查进程内的UBODTManager
2. 如果没有缓存，重新加载UBODT

### 如何让FMM/CMM使用Daemon的UBODT？

有两种方案：

#### 方案1：共享内存（需要实现）

修改UBODTManager，使其能够：
1. 检测daemon是否在运行
2. 通过共享内存或socket连接到daemon
3. 直接使用daemon中的UBODT数据

#### 方案2：使用Unix Socket（推荐）

创建一个轻量级的客户端库，让FMM/CMM通过socket与daemon通信。

### 临时解决方案

在daemon实现客户端连接之前，你可以：

1. **保持daemon运行**（为将来的功能做准备）
2. **使用多线程并行处理**：
   ```bash
   # 启动daemon
   ubodt_daemon start --ubodt data/ubodt.bin

   # 在同一进程中并行处理多个文件
   # 需要编写自定义脚本或修改FMMApp支持批处理
   ```

## Daemon管理

### 检查Daemon是否在运行

```bash
# 方法1：使用status命令
ubodt_daemon status

# 方法2：检查进程
ps aux | grep ubodt_daemon

# 方法3：检查PID文件
cat /tmp/ubodt_daemon.pid
```

### 停止无响应的Daemon

```bash
# 方法1：使用stop命令
ubodt_daemon stop

# 方法2：手动发送信号
kill -TERM $(cat /tmp/ubodt_daemon.pid)

# 方法3：强制杀死（最后手段）
kill -9 $(cat /tmp/ubodt_daemon.pid)
rm /tmp/ubodt_daemon.pid
rm /tmp/ubodt_daemon_status.txt
```

### 自动启动Daemon

#### 使用systemd（Linux）

创建服务文件 `/etc/systemd/system/ubodt-daemon.service`：

```ini
[Unit]
Description=UBODT Daemon
After=network.target

[Service]
Type=simple
User=dell
WorkingDirectory=/home/dell/fmm_sjtugnc
ExecStart=/home/dell/miniconda3/bin/ubodt_daemon start --ubodt /path/to/ubodt.bin
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable ubodt-daemon
sudo systemctl start ubodt-daemon
sudo systemctl status ubodt-daemon
```

#### 使用screen/tmux

```bash
# 启动daemon在screen会话中
screen -dmS ubodt_daemon bash -c "ubodt_daemon start --ubodt data/ubodt.bin"

# 查看daemon输出
screen -r ubodt_daemon

# 分离会话：Ctrl+A, D
```

## 常见问题

### Q: 为什么在另一个终端看不到UBODT？

A: 因为`ubodt_manager load`只在当前进程中加载，进程退出后UBODT就释放了。使用`ubodt_daemon`可以保持UBODT持久化。

### Q: Daemon占多少内存？

A: 大约等于UBODT文件大小加上几十MB的程序开销。例如：
- 1GB的UBODT文件 → Daemon约占用1.05GB内存

### Q: 可以同时运行多个daemon吗？

A: 当前实现不支持。如果需要加载多个不同的UBODT，可以考虑实现多daemon支持。

### Q: Daemon崩溃了怎么办？

A: 可以使用systemd或supervisor配置自动重启。或者手动：
```bash
ubodt_daemon stop
ubodt_daemon start --ubodt data/ubodt.bin
```

### Q: 如何查看daemon的日志？

A: Daemon使用spdlog输出到stdout/stderr。如果使用screen或systemd，可以重定向到文件：
```bash
ubodt_daemon start --ubodt data/ubodt.bin > daemon.log 2>&1 &
```

## 性能优势

### 传统方式

```
运行1: 加载UBODT(5s) + 处理(10s) = 15s
运行2: 加载UBODT(5s) + 处理(10s) = 15s
运行3: 加载UBODT(5s) + 处理(10s) = 15s
总计: 45s
```

### 使用Daemon

```
启动Daemon: 加载UBODT(5s)
运行1: 处理(10s)
运行2: 处理(10s)
运行3: 处理(10s)
总计: 5s + 30s = 35s
节省: 10s (22%)
```

如果处理100个文件：
- 传统: 1500s
- Daemon: 1005s
- 节省: 495s (8.25分钟)

## 安全注意事项

1. **PID文件位置**: 默认在`/tmp/`，多用户系统可能需要修改为私有目录
2. **权限**: 确保只有授权用户能启动/停止daemon
3. **资源限制**: 大型UBODT可能消耗大量内存，注意系统资源

## 未来改进

- [ ] 实现客户端-服务器架构，让FMM/CMM直接连接daemon
- [ ] 支持同时加载多个UBODT
- [ ] 添加LRU缓存策略，自动管理多个UBODT
- [ ] 支持动态重新加载UBODT（无需重启daemon）
- [ ] 添加Web界面管理
- [ ] 支持远程UBODT服务

## 总结

**UBODT Daemon** 提供了一个持久化的UBODT存储服务，特别适合：

✅ 批量处理大量轨迹文件
✅ 需要频繁进行地图匹配的服务
✅ 多用户/多进程共享UBODT
✅ 减少重复加载时间

虽然当前版本的FMM/CMM还不能直接使用daemon中的UBODT，但daemon架构为将来的功能扩展奠定了基础。
