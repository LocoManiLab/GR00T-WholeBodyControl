#!/usr/bin/env bash
# fix_zed.sh —— 强制让 ZED-M 在 USB3 上重新枚举出视频接口。
# 只操作 ZED 所在端口 (1-3.3 / hub 1-3 port 3)，绝不触碰 WiFi(1-2.4) 这条 SSH 生命线。
# 用法: sudo bash ~/fix_zed.sh
set -u

ZED_DEV="1-3.3"     # ZED sysfs 设备
ZED_HUB="1-3"       # 所在 hub
ZED_PORT="3"        # hub 上的端口号
SVC="composed_camera_server.service"

say(){ echo -e "\n=== $* ==="; }

zed_state(){
  echo -n "  lsusb: "; lsusb | grep -i stereolabs || echo "(无 STEREOLABS)"
  for d in /sys/bus/usb/devices/*; do
    [ "$(cat "$d/idVendor" 2>/dev/null)" = "2b03" ] && \
      echo "  $d  speed=$(cat "$d/speed" 2>/dev/null)M  product=$(cat "$d/product" 2>/dev/null)"
  done
  echo -n "  视频接口(UVC)? "; lsusb -t 2>/dev/null | grep -iE "uvcvideo" >/dev/null && echo "有 ✅" || echo "无 ❌"
  echo -n "  5000M 链路? ";    lsusb -t 2>/dev/null | grep -q "5000M" && echo "有 5000M 端口" || echo "未见 5000M"
}

say "[0] 复位前状态"; zed_state

say "[1] 停掉相机服务，释放 ZED"
systemctl stop "$SVC" 2>/dev/null
sleep 2; pkill -f composed_camera 2>/dev/null; sleep 1
pgrep -af composed_camera >/dev/null && echo "  ⚠️ 仍有进程占用" || echo "  已释放"

say "[2] 方法A: uhubctl 给 ZED 端口断电冷重启 (最有效)"
if uhubctl -l "$ZED_HUB" -p "$ZED_PORT" -a cycle -d 2 2>&1; then
  echo "  uhubctl cycle 成功"
else
  echo "  ✗ uhubctl 不支持该 hub 电源开关，转方法B"
  say "[2b] 方法B: authorized 重枚举"
  if [ -e "/sys/bus/usb/devices/$ZED_DEV/authorized" ]; then
    echo 0 > "/sys/bus/usb/devices/$ZED_DEV/authorized"; sleep 2
    echo 1 > "/sys/bus/usb/devices/$ZED_DEV/authorized"; echo "  authorized 0->1 done"
  else
    echo "  $ZED_DEV 不存在(可能已掉)，尝试 usbreset"
    BUSDEV=$(lsusb | grep -i stereolabs | sed -E 's#Bus ([0-9]+) Device ([0-9]+):.*#/dev/bus/usb/\1/\2#')
    [ -n "$BUSDEV" ] && usbreset "$BUSDEV"
  fi
  say "[2c] 方法C: 复位父 hub $ZED_HUB (会顺带重连 G1 网线，不影响本 SSH)"
  if [ -e "/sys/bus/usb/devices/$ZED_HUB/authorized" ]; then
    echo 0 > "/sys/bus/usb/devices/$ZED_HUB/authorized"; sleep 2
    echo 1 > "/sys/bus/usb/devices/$ZED_HUB/authorized"; echo "  hub 重枚举 done"
  fi
fi

echo "  等待重新枚举 ..."; sleep 5

say "[3] 复位后状态"; zed_state

say "[4] 重启相机服务"
systemctl start "$SVC" 2>/dev/null && echo "  服务已启动" || echo "  ⚠️ 服务启动失败"
echo ""
echo ">>> 若上面显示 speed=5000M 且 '视频接口(UVC) 有 ✅' = 修复成功。"
echo ">>> 若仍是 12M / 只有 HID = 软件复位无效，基本可判定是线材/接口接触问题，"
echo ">>>    请换一根 ZED 原装 USB3 线、或换一个机身 USB3(蓝口)再跑一次本脚本。"
