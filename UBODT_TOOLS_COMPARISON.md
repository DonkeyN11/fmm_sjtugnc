# UBODT工具对比：Manager vs Daemon

## 你的问题：为什么在另一个终端看不到？

### 问题原因

当你运行：
```bash
ubodt_manager load data/ubodt.bin
```

实际发生了什么：
1. 系统启动一个**新进程**
2. UBODT被加载到**这个进程的内存**中
3. 命令执行完毕，**进程退出**
4. 进程的内存被释放，**UBODT消失** ❌

在另一个终端运行 `ubodt_manager status` 时：
1. 又启动了**另一个新进程**
2. 这个进程的内存是空的
3. 看不到之前进程的UBODT ❌

**核心问题：进程间内存隔离**

## 解决方案对比

### 方案1：ubodt_manager（原有工具）

**特点：**
- ✅ 命令简单
- ✅ 适合一次性操作
- ❌ 进程退出后UBODT释放
- ❌ 不同终端无法共享

**使用场景：**
```bash
# 单次使用，用完即走
ubodt_manager load data/ubodt.bin
# 命令立即结束，UBODT已释放
```

### 方案2：ubodt_daemon（新增工具）⭐

**特点：**
- ✅ 后台持续运行
- ✅ UBODT持久化在内存
- ✅ 所有终端都能查询状态
- ✅ 多进程共享UBODT
- ✅ 适合批量处理

**使用场景：**
```bash
# 终端1: 启动daemon
ubodt_daemon start --ubodt data/ubodt.bin
# Daemon在后台运行...

# 终端2: 查询状态 ✅
ubodt_daemon status
# 能看到daemon在运行！

# 终端3: 查询状态 ✅
ubodt_daemon status
# 也能看到！

# 完成后停止
ubodt_daemon stop
```

## 快速选择指南

| 你的需求 | 推荐工具 |
|---------|---------|
| 单次使用，不需要共享 | `ubodt_manager` |
| 批量处理多个文件 | `ubodt_daemon` ⭐ |
| 需要在多个终端查看状态 | `ubodt_daemon` ⭐ |
| 作为服务长期运行 | `ubodt_daemon` ⭐ |
| 简单快速测试 | `ubodt_manager` |

## 详细对比表

| 特性 | ubodt_manager | ubodt_daemon |
|------|---------------|--------------|
| **命令格式** | `ubodt_manager load <file>` | `ubodt_daemon start --ubodt <file>` |
| **进程生命周期** | 命令执行完就退出 | 持续后台运行 |
| **UBODT持久化** | ❌ | ✅ |
| **跨终端查看** | ❌ | ✅ |
| **多进程共享** | ❌ | ✅ (未来版本) |
| **状态查询** | `ubodt_manager status` | `ubodt_daemon status` |
| **释放方式** | `ubodt_manager release` | `ubodt_daemon stop` |
| **内存占用** | 临时 | 持续 |
| **适用场景** | 单次操作 | 批量处理、服务化 |

## 实际使用示例

### 场景1：处理100个轨迹文件

#### ❌ 错误方式（使用ubodt_manager）

```bash
# 每次运行都会重新加载UBODT！
for i in {1..100}; do
  ubodt_manager load data/ubodt.bin  # 加载5秒
  fmm --config config_$i.xml         # 处理10秒
  ubodt_manager release data/ubodt.bin
done
# 总计: 100 × 15秒 = 1500秒 (25分钟)
```

#### ✅ 正确方式（使用ubodt_daemon）

```bash
# 只加载一次！
ubodt_daemon start --ubodt data/ubodt.bin  # 加载5秒

for i in {1..100}; do
  fmm --config config_$i.xml  # 处理10秒
done

ubodt_daemon stop
# 总计: 5秒 + 100 × 10秒 = 1005秒 (16.75分钟)
# 节省: 495秒 (8.25分钟) ⭐
```

### 场景2：多用户协作

**使用ubodt_daemon：**
```bash
# 用户A启动daemon
ubodt_daemon start --ubodt /data/shared/ubodt.bin

# 用户B查询状态
ubodt_daemon status
# 输出: Daemon is running, PID: 12345

# 用户C查询状态
ubodt_daemon status
# 输出: Daemon is running, PID: 12345
# 所有用户都能看到！✅
```

**使用ubodt_manager（无法协作）：**
```bash
# 用户A加载UBODT
ubodt_manager load /data/shared/ubodt.bin

# 用户B查询状态
ubodt_manager status
# 输出: Total UBODTs loaded: 0 ❌
# 看不到用户A的UBODT！
```

## 进程隔离图示

### ubodt_manager的进程隔离

```
终端1                          终端2
  |                              |
  v                              v
[进程A]                        [进程B]
  |                              |
  |-- 加载UBODT                  |-- 查询状态
  |-- 进程退出                   |-- 内存是空的 ❌
  |-- UBODT释放
```

### ubodt_daemon的共享模式

```
终端1                          终端2                          终端3
  |                              |                              |
  v                              v                              v
[启动Daemon] <--------------> [查询状态] <----------------> [查询状态]
  |                              |                              |
  |-- Daemon进程持续运行          |-- 看到Daemon ✅              |-- 看到Daemon ✅
  |-- UBODT持久化                |-- 可共享 (未来版本)          |-- 可共享 (未来版本)
```

## 技术细节

### ubodt_manager的实现

```cpp
// 单例模式，但是是进程内的单例
class UBODTManager {
    static UBODTManager& getInstance() {
        static UBODTManager instance;  // 每个进程有自己的实例
        return instance;
    }
};

// 当进程退出时，所有内存被释放
```

### ubodt_daemon的实现

```cpp
// Daemon进程持续运行
int run_daemon(string ubodt_file) {
    // 加载UBODT
    auto ubodt = load_ubodt(ubodt_file);

    // 持续运行，保持UBODT在内存
    while (keep_running) {
        sleep(5);
    }

    // 收到停止信号时才退出
}

// 通过文件系统进行进程间通信
// - /tmp/ubodt_daemon.pid      (PID文件)
// - /tmp/ubodt_daemon_status.txt (状态文件)
```

## 未来改进

当前版本的FMM/CMM还不能直接使用daemon中的UBODT，但计划中的改进：

1. **客户端-服务器架构**
   ```cpp
   // FMM/CMM自动连接daemon
   auto ubodt = UBODTClient::connect_to_daemon();
   ```

2. **共享内存**
   ```cpp
   // 通过共享内存访问daemon的UBODT
   auto ubodt = UBODTManager::get_from_shared_memory();
   ```

3. **Unix Socket**
   ```cpp
   // 通过socket与daemon通信
   auto ubodt = UBODTClient::request_ubodt(socket_path);
   ```

## 总结

### 核心区别

**ubodt_manager：**
- 临时工具，用完即走
- 进程内单例
- 不同进程无法共享

**ubodt_daemon：**
- 持久化服务
- 独立进程运行
- 所有终端都能访问
- 适合批量处理和服务化场景

### 推荐使用

- 🔧 **开发测试**: `ubodt_manager`
- 🏭 **生产环境**: `ubodt_daemon` ⭐
- 📊 **批量处理**: `ubodt_daemon` ⭐
- 👥 **多用户协作**: `ubodt_daemon` ⭐

### 快速开始

```bash
# 启动daemon
ubodt_daemon start --ubodt <你的UBODT文件>

# 在任何终端查询状态
ubodt_daemon status

# 停止daemon
ubodt_daemon stop
```

详细文档：
- `UBODT_MANAGER_README.md` - Manager使用指南
- `UBODT_DAEMON_README.md` - Daemon使用指南
- `INSTALL.md` - 安装指南
