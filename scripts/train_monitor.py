"""
MiniMind 训练实时监控面板（零侵入：只读日志文件，不影响训练进程）

用法:
    cd scripts && ../.venv/bin/python -m streamlit run train_monitor.py --server.port 8502

支持解析:
    - 预训练/SFT/LoRA/蒸馏:  Epoch:[1/1](200/39695), loss: 6.28, ... lr: 0.0005, epoch_time: 756.0min
    - DPO:                   ... loss: 0.59, dpo_loss: 0.59, ... learning_rate: 1e-07
    - PPO/GRPO/AgentRL:      ... Reward: -1.52, KL: 0.001, ... Actor Loss: 0.14
"""
import re
import time
import glob
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

st.set_page_config(page_title="MiniMind 训练监控", page_icon="📈", layout="wide")

# ============ 日志解析 ============
STEP_RE = re.compile(r"Epoch:\[(\d+)/(\d+)\]\((\d+)/(\d+)\)")
NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
FIELD_PATTERNS = {
    "loss": re.compile(rf"\bloss:\s*{NUM}", re.I),
    "logits_loss": re.compile(rf"logits_loss:\s*{NUM}", re.I),
    "aux_loss": re.compile(rf"aux_loss:\s*{NUM}", re.I),
    "dpo_loss": re.compile(rf"dpo_loss:\s*{NUM}", re.I),
    "distill": re.compile(rf"distill:\s*{NUM}", re.I),
    "ce": re.compile(rf"\bce:\s*{NUM}", re.I),
    "reward": re.compile(rf"Reward:\s*{NUM}", re.I),
    "kl": re.compile(rf"KL(?:_ref)?:\s*{NUM}", re.I),
    "actor_loss": re.compile(rf"Actor Loss:\s*{NUM}", re.I),
    "critic_loss": re.compile(rf"Critic Loss:\s*{NUM}", re.I),
    "avg_len": re.compile(rf"Avg(?:erage)? Response Len:\s*{NUM}|AvgLen:\s*{NUM}", re.I),
    "lr": re.compile(rf"(?:learning_rate|lr|LR):\s*{NUM}", re.I),
    "eta_min": re.compile(rf"epoch_time:\s*{NUM}min", re.I),
}


def parse_log(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = STEP_RE.search(line)
                if not m:
                    continue
                row = {
                    "epoch": int(m.group(1)),
                    "epochs": int(m.group(2)),
                    "step": int(m.group(3)),
                    "total": int(m.group(4)),
                }
                for name, pat in FIELD_PATTERNS.items():
                    fm = pat.search(line)
                    if fm:
                        val = next((g for g in fm.groups() if g is not None), None)
                        if val is not None:
                            try:
                                row[name] = float(val)
                            except ValueError:
                                pass
                rows.append(row)
    except FileNotFoundError:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 跨 epoch 的全局步数，便于连续绘图
    df["global_step"] = (df["epoch"] - 1).clip(lower=0) * df["total"] + df["step"]
    return df


def fmt_eta(minutes):
    if minutes is None or pd.isna(minutes):
        return "—"
    td = timedelta(minutes=float(minutes))
    h, rem = divmod(int(td.total_seconds()), 3600)
    return f"{h}h {rem // 60}m"


# ============ 侧边栏 ============
st.sidebar.title("⚙️ 设置")
logs = sorted(glob.glob("/tmp/mm_*.log"), key=os.path.getmtime, reverse=True)
custom = st.sidebar.text_input("自定义日志路径", "")
options = ([custom] if custom else []) + logs
if not options:
    st.warning("未找到日志文件（默认扫描 /tmp/mm_*.log）")
    st.stop()

log_path = st.sidebar.selectbox(
    "日志文件（按修改时间排序）",
    options,
    format_func=lambda p: f"{os.path.basename(p)}  ({datetime.fromtimestamp(os.path.getmtime(p)):%m-%d %H:%M})"
    if os.path.exists(p) else p,
)
auto = st.sidebar.checkbox("自动刷新", value=True)
interval = st.sidebar.slider("刷新间隔（秒）", 5, 120, 20)
smooth = st.sidebar.slider("曲线平滑窗口", 1, 50, 5)
tail_n = st.sidebar.slider("原始日志显示行数", 5, 60, 15)

# ============ 主体 ============
st.title("📈 MiniMind 训练监控")
st.caption(f"数据源：`{log_path}` ｜ 最后刷新：{datetime.now():%H:%M:%S}")

df = parse_log(log_path)
if df.empty:
    st.info("日志中还没有可解析的训练记录，等待第一个 log_interval 输出…")
    if auto:
        time.sleep(interval)
        st.rerun()
    st.stop()

last = df.iloc[-1]
progress = last["step"] / last["total"] if last["total"] else 0
running = os.path.getmtime(log_path) > time.time() - 300

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("进度", f"{progress * 100:.1f}%", f"{int(last['step'])} / {int(last['total'])} 步")
main_metric = "reward" if "reward" in df.columns else "loss"
if main_metric in df.columns:
    delta = float(df[main_metric].iloc[-1] - df[main_metric].iloc[0])
    c2.metric(f"当前 {main_metric}", f"{last[main_metric]:.4f}", f"{delta:+.4f} 自起点")
c3.metric("剩余时间", fmt_eta(last.get("eta_min")))
c4.metric("学习率", f"{last['lr']:.2e}" if "lr" in df.columns else "—")
c5.metric("状态", "🟢 训练中" if running else "⚪️ 已停止", f"Epoch {int(last['epoch'])}/{int(last['epochs'])}")

st.progress(min(progress, 1.0))

# 曲线
plot_cols = [c for c in ["loss", "logits_loss", "ce", "distill", "dpo_loss", "reward",
                         "actor_loss", "critic_loss", "kl", "aux_loss"] if c in df.columns]
if plot_cols:
    picked = st.multiselect("曲线指标", plot_cols, default=plot_cols[:2])
    if picked:
        chart_df = df.set_index("global_step")[picked]
        if smooth > 1:
            chart_df = chart_df.rolling(smooth, min_periods=1).mean()
        st.line_chart(chart_df, height=340)

col_a, col_b = st.columns(2)
if "lr" in df.columns:
    col_a.caption("学习率")
    col_a.line_chart(df.set_index("global_step")[["lr"]], height=200)
if "avg_len" in df.columns:
    col_b.caption("平均生成长度")
    col_b.line_chart(df.set_index("global_step")[["avg_len"]], height=200)
elif "eta_min" in df.columns:
    col_b.caption("剩余时间估计（分钟）")
    col_b.line_chart(df.set_index("global_step")[["eta_min"]], height=200)

with st.expander(f"原始日志（末尾 {tail_n} 行）"):
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            st.code("".join(f.readlines()[-tail_n:]), language="text")
    except Exception as e:  # noqa: BLE001
        st.error(str(e))

if auto:
    time.sleep(interval)
    st.rerun()
