# 通过 Tailscale 实现 G1 + PICO 公网远距离遥操

让机器人和 PICO **各自连不同的 WiFi（不同局域网 / 公网）** 也能遥操。原理：用
[Tailscale](https://tailscale.com/)（基于 WireGuard 的零配置组网）把"运行 PICO
streamer 的主机"和"机器人机载板卡"组进同一个虚拟局域网，启动时把对端地址填成
Tailscale 地址即可。

> **本目录不修改任何仓库原有代码**，全部通过现有命令行参数实现。

---

## 1. 它做了什么（网络拓扑）

```
┌─ 操作者侧（本地 WiFi，低延迟，不走公网）──┐   ┌─ 机器人侧（本地，不走公网）──────────┐
│ PICO 头显                                   │   │ Jetson 机载板卡                      │
│   ↕ 本地 WiFi：PICO app 填主机【LAN IP】    │   │   跑 C++ sonic 部署 (deploy.sh)      │
│ 你的主机 (x86_64)                           │   │   + 相机服务端 (camera server)       │
│   跑 PICO streamer  PUB :5556               │   │   ↕ 网线 192.168.123.x (SDK/DDS)     │
│   跑 camera viewer  SUB → 机器人            │   │ G1 机器人                            │
│   [Tailscale 节点: operator-host]           │   │   [Tailscale 节点: g1-robot]         │
└──────────────────┬──────────────────────────┘   └──────────────┬───────────────────────┘
                   └────────── Tailscale / WireGuard（公网）──────┘
        · 控制流：机器人 deploy 端 SUB connect → operator-host:5556
        · 视频流：主机 viewer SUB connect → g1-robot:5555
        · SDK/DDS 链路全程留在机器人本地（绝不上公网）
```

**关键设计**：两条延迟敏感的本地链路保持本地——
- **PICO ↔ 主机** 走本地 WiFi（不要让 PICO 直连远端）；
- **deploy ↔ 机器人 SDK/DDS** 走机载板卡到 G1 的网线（DDS 高频控制无法上公网）。

只有 **ZMQ 姿态控制流** 和 **视频回传** 两条穿 Tailscale。

---

## 2. 前置条件

**机器人机载板卡（Jetson/arm64）**
- 已 build 好 `gear_sonic_deploy/`（`cd gear_sonic_deploy && just build`）。
- 已装相机 venv：`bash install_scripts/install_camera_server.sh`（产生 `.venv_camera`）。
- 网线连到 G1 的 `192.168.123.x` 网络。
- 能联网安装 Tailscale。

**你的主机（x86_64）**
- 已装遥操 venv：`bash install_scripts/install_pico.sh`（产生 `.venv_teleop`）。
- 已装相机 venv：`bash install_scripts/install_camera_server.sh`（产生 `.venv_camera`，用于看视频）。
- PICO 与主机在**同一本地 WiFi**，XRoboToolKit PC Service 已就绪。

---

## 3. 一次性组网（装 Tailscale）

1. 注册 Tailscale 账号，在管理后台
   [Settings → Keys](https://login.tailscale.com/admin/settings/keys) 生成一个
   **auth key**（建议勾选 *Reusable*，可选打 tag）。

2. **机器人**（无头，用 auth key）：
   ```bash
   TS_AUTHKEY=tskey-xxxxxxxx bash remote_teleop/setup_tailscale.sh --hostname g1-robot
   ```

3. **主机**（可交互浏览器登录，也可同样用 auth key）：
   ```bash
   bash remote_teleop/setup_tailscale.sh --hostname operator-host
   ```

4. 验证两节点互通（任一端执行）：
   ```bash
   tailscale ping g1-robot
   tailscale ping operator-host
   ```
   - 看到 `pong ... via 100.x` → **P2P 直连，延迟低（理想）**。
   - 看到 `via DERP` → 走了中继，延迟更高，能用但建议排查（NAT/防火墙）。

> 默认用 Tailscale 的 **MagicDNS 名**（`g1-robot` / `operator-host`）作为对端地址。
> 若名字解析有问题，改用 `tailscale ip -4` 打印出的 `100.x` IP 即可（所有脚本都支持传 IP）。

---

## 4. 启动顺序（远程真机，共 4 个终端）

> ⚠️ 上真机前请先读 [第 6 节安全须知](#6-安全须知必读)，并建议先按 [第 5 节](#5-上真机前先在-sim-验证) 隔着 Tailscale 在 sim 验证一遍。

**机器人终端 A —— 相机服务端**
```bash
bash remote_teleop/robot_camera.sh
```

**机器人终端 B —— C++ 部署（连主机 streamer）**
```bash
bash remote_teleop/robot_deploy.sh real
# 等待出现 "Init done"
# 若 MagicDNS 名不通，可改用主机的 Tailscale IP：
#   bash remote_teleop/robot_deploy.sh 100.x.y.z real
```

**主机终端 C —— PICO streamer**
```bash
bash remote_teleop/host_streamer.sh --vis_vr3pt --vis_smpl
# 按打印提示，把【本地 LAN IP】填进 PICO 的 XRoboToolKit app
```

**主机终端 D —— 视频 viewer**
```bash
bash remote_teleop/host_viewer.sh
# 若 MagicDNS 名不通：bash remote_teleop/host_viewer.sh 100.x.y.z
```

随后按仓库原有 VR 遥操流程操作（详见
[`docs/source/tutorials/vr_wholebody_teleop.md`](../docs/source/tutorials/vr_wholebody_teleop.md)）：

1. 站标定姿态（直立、双脚并拢、上臂垂下、小臂前屈 90°、掌心向内）。
2. PICO 上 **A+B+X+Y** 启动策略并标定。
3. **A+X** 进入 POSE 全身遥操；再次 **A+X** 回 PLANNER。
4. **A+B+X+Y** 停止。

---

## 5. 上真机前：先在 sim 验证

强烈建议先隔着 Tailscale 把控制链路在仿真里跑通：

```bash
# 主机：起 MuJoCo 仿真（终端1）
source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py

# 主机：起 PICO streamer（终端2）
bash remote_teleop/host_streamer.sh --vis_vr3pt --vis_smpl

# 机器人/或任意第二台机：起部署连主机（终端3）
bash remote_teleop/robot_deploy.sh sim
```

部署端打印 `ZMQ STREAMING MODE: ENABLED` 且能看到 motion 帧到达，即说明
streamer → deploy 的姿态流已经跨 Tailscale 跑通。

---

## 6. 安全须知（必读）

- **⚠️ 公网断流不会自动急停。** 部署端代码里没有"流中断看门狗"——一旦
  Tailscale 链路断开，POSE 模式下机器人会**冻结在最后姿态**，而不是软急停。
  务必：
  - 机器人旁**留一名操作员**，能在部署终端（终端 B）按 **`O`** 立即急停；
  - 不要把 PICO 的 **A+B+X+Y** 急停当作唯一手段——该信号本身也要穿公网才能到达部署端。
- **先 sim 后真机**，确认延迟/抖动可接受再上真机。WBC 全身平衡对抖动敏感。
- **优先 P2P 直连**：`tailscale ping` 出现 `via DERP` 说明走中继，延迟偏高，尽量排查到直连。
- 上真机前**单独演练一次断网**（拔主机网线 / 关 streamer），观察机器人行为并确认急停预案可用。

---

## 7. 排障

| 现象 | 排查 |
|---|---|
| `tailscale ping` 不通 | 两端是否都 `setup_tailscale.sh` 入网；`tailscale status` 是否互相可见 |
| 走 DERP 中继延迟高 | 检查 NAT/防火墙；尝试开放 UDP 41641；企业网络可能需要 DERP |
| 对端连不上 5556 / 5555 | 监听端机器是否拦了 tailscale0 入站：`sudo ufw allow in on tailscale0` |
| MagicDNS 名解析失败 | 改用 `tailscale ip -4` 的 `100.x` IP；确认 `tailscale up` 带了 `--accept-dns=true` |
| 部署端收不到姿态 | 确认 streamer 在主机已起、PICO 已开始 stream；`--zmq-host` 是否填的是主机地址 |
| 看不到视频 | `robot_camera.sh` 是否在机器人已起；`host_viewer.sh` 的地址是否指向机器人 |

**端口**：控制流 `5556`、视频流 `5555`，均可在对应脚本参数中覆盖。

---

## 8. 脚本清单

| 脚本 | 运行位置 | 作用 |
|---|---|---|
| `setup_tailscale.sh` | 两台都跑 | 安装 + 加入 Tailscale（幂等） |
| `robot_deploy.sh` | 机器人 | 包装 `deploy.sh`，`--zmq-host` 指向主机 |
| `robot_camera.sh` | 机器人 | 启动相机服务端（视频源） |
| `host_streamer.sh` | 主机 | 启动 PICO streamer，并打印待填地址 |
| `host_viewer.sh` | 主机 | 启动视频 viewer，连机器人相机 |
