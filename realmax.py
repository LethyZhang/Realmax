"""

# 假设你已经有了 model 对象
import realmax # 假设上面的文件名叫 realmax.py

# 1. 最简调用
mp = realmax.msrr(model)
print(f"Result: {mp} MP")

# 2. 指定对齐倍数
mp = realmax.msrr(model, size_multiple=8)

# 3. 开启 Debug 查看搜索过程
mp = realmax.msrr(model, debug=True)

# 4. 调用Score计算
mp_score = realmax.score(net, size_multiple=32, debug=True)

# 5. 调用Average FPS计算
mp_ave_fps = realmax.ave_fps(net, size_multiple=32, debug=True)

"""



import os
import cv2
import glob
import math
import argparse
import importlib.util
import sys
import gc
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.serialization
from thop import profile
from thop.vision.basic_hooks import zero_ops
from thop.vision import basic_hooks
import torch.nn.functional as F
import time


# =====================================================
#  初始工具
# =====================================================
custom_ops = {
    nn.PReLU: zero_ops
}
anchors = [192, 256, 360, 480, 512, 720, 900, 1000, 1080, 1200, 1440, 1600, 1800, 2160, 4320, 8640, 17280, 34560]


#   start
#   source /home_ext/zls-uestc-tmp/.virtualenvs/PyCharmMiscProject/bin/activate
#   cd Realmax
#   CUDA_VISIBLE_DEVICES=4 python main.py -device cuda


# =====================================================
#  CUDA 清理工具
# =====================================================
def cleanup_cuda(tag=""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if tag:
        print(f"[cleanup] {tag} done.")


# =====================================================
#  参数解析
# =====================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ('yes', 'true', 't', '1'):
        return True
    elif v in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




# =====================================================
# 通用工具
# =====================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def align_down(x: int, m: int) -> int:
    if m <= 1:
        return int(x)
    return max(m, (int(x) // m) * m)


def log(f, s: str):
    if f is None:
        return

    #如果是API Logger，根据debug状态决定是否print
    if hasattr(f, 'is_api_logger'):
        if f.debug:
            print(str(s))
    else:
        # 既不是None也不是API Logger，说明是正常文件模式，照常打印和写入
        print(s)
        f.write(str(s) + "\n")
        f.flush()


@torch.no_grad()
def global_warmup(net, device, fps_sample, size_wh=(256, 256),
                  min_frames=50, max_seconds=10.0, multiple=1):
    w, h = size_wh
    w = align_down(w, multiple)
    h = align_down(h, multiple)
    img = prepare_fps_input(fps_sample, device, (w, h))


    # CUDA 先同步一下，避免计时与 allocator 抖动
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n = 0
    while n < min_frames:
        _ = net(img)
        n += 1
        if (time.perf_counter() - t0) >= max_seconds:
            break

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# 修改后的 prepare_fps_input 函数
def prepare_fps_input(fps_sample, device, size_wh):
    # 修改点：如果传入 None，生成随机噪点图（Random Noise），
    # 避免全黑图（全0）可能触发的 GPU/算法层面的稀疏计算优化，保证测试公平性。
    if fps_sample is None:
        w, h = size_wh
        # 生成 0-255 之间的随机整数，形状为 (h, w, 3)
        rand_arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(rand_arr)
    else:
        img = Image.open(fps_sample).convert("RGB")

    # 保持原有的 resize 逻辑（如果是随机图，这一步其实是原样 resize，不影响逻辑）
    img = img.resize(size_wh, Image.BICUBIC)
    # ===================================================

    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    return t

def is_oom_error(e: Exception) -> bool:
    s = str(e).lower()
    return ("out of memory" in s) or ("cuda error: out of memory" in s)

@torch.no_grad()
def safe_measure(net, img, loops, warmup, return_jitter):
    """
    包装 measure_fps_jitter：遇到 OOM 不崩溃，返回 (None, None, True)
    """
    try:
        fps, jit = measure_fps_jitter(net, img, loops=loops, warmup=warmup, return_jitter=return_jitter)
        return fps, jit, False
    except RuntimeError as e:
        if is_oom_error(e):
            cleanup_cuda("OOM")
            return None, None, True
        raise


# =====================================================
# 模型加载（仅加载结构，不加载权重）
# =====================================================
def load_model(device, model_name, model_path):

    spec = importlib.util.spec_from_file_location("model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["model"] = module

    if not hasattr(module, model_name):
        raise AttributeError(f"❌ 模型文件中没有类: {model_name}")
    ModelClass = getattr(module, model_name)

    print("⚠️ 未加载任何权重，仅使用随机初始化模型进行 FPS / FLOPs 测试")
    net = ModelClass().to(device)

    net.eval()
    return net


# =====================================================
# FLOPs
# =====================================================
def compute_flops_cpu(model, h, w):
    dummy = torch.randn(1, 3, h, w)
    flops, params = profile(model, inputs=(dummy,), custom_ops=custom_ops)
    del dummy
    return flops, params


# =====================================================
# FPS & jitter
# =====================================================
@torch.no_grad()
def measure_fps_jitter(net, img, loops=120, warmup=50, return_jitter=True):
    """
    不逐次 synchronize 的 CUDA 计时版本：
      - CUDA: 每次迭代只 record event，最后统一 synchronize
      - CPU : 仍用 perf_counter

    return_jitter:
      - True : 返回 (fps, jitter%)
      - False: 返回 (fps, None)
    """
    is_cuda = (img.device.type == "cuda") and torch.cuda.is_available()

    # warmup（不计时）
    for _ in range(warmup):
        _ = net(img)

    if is_cuda:
        # 清空 CUDA 队列，避免前面操作干扰
        torch.cuda.synchronize()

        # 为每次迭代创建一对 events（loops<=200 这种规模完全可接受）
        starters = [torch.cuda.Event(enable_timing=True) for _ in range(loops)]
        enders   = [torch.cuda.Event(enable_timing=True) for _ in range(loops)]

        for i in range(loops):
            starters[i].record()
            _ = net(img)
            enders[i].record()

        # 只在最后同步一次
        torch.cuda.synchronize()

        times_ms = np.empty(loops, dtype=np.float64)
        for i in range(loops):
            times_ms[i] = starters[i].elapsed_time(enders[i])

    else:
        times = []
        for _ in range(loops):
            t0 = time.perf_counter()
            _ = net(img)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times_ms = np.array(times, dtype=np.float64)

    mean_ms = float(times_ms.mean())
    if mean_ms <= 0:
        return float("inf"), (0.0 if return_jitter else None)

    fps = 1000.0 / mean_ms

    if not return_jitter:
        return fps, None

    std_ms = float(times_ms.std())
    jitter = (std_ms / mean_ms) * 100.0 if mean_ms > 0 else 0.0
    return fps, jitter


@torch.no_grad()
def measure_precision_p95(net, img, loops=100, warmup=20):
    """
    不逐次 synchronize 的 P95 计时：
      - CUDA: 每次迭代 record 一对 events，最后统一 synchronize，然后算每次 elapsed_time
      - CPU : perf_counter 每次测一次（CPU 没异步队列问题）
    """
    # warmup（不计时）
    for _ in range(warmup):
        _ = net(img)

    is_cuda = (img.device.type == "cuda") and torch.cuda.is_available()

    if is_cuda:
        torch.cuda.synchronize()

        starters = [torch.cuda.Event(enable_timing=True) for _ in range(loops)]
        enders   = [torch.cuda.Event(enable_timing=True) for _ in range(loops)]

        for i in range(loops):
            starters[i].record()
            _ = net(img)
            enders[i].record()

        torch.cuda.synchronize()

        times_ms = np.empty(loops, dtype=np.float64)
        for i in range(loops):
            times_ms[i] = starters[i].elapsed_time(enders[i])

    else:
        times = []
        for _ in range(loops):
            t0 = time.perf_counter()
            _ = net(img)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        times_ms = np.array(times, dtype=np.float64)

    mean_ms = float(times_ms.mean())
    std_ms  = float(times_ms.std())
    p95_ms  = float(np.percentile(times_ms, 95))

    mean_fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
    p95_fps  = 1000.0 / p95_ms if p95_ms > 0 else float("inf")
    jitter   = (std_ms / mean_ms) * 100.0 if mean_ms > 0 else 0.0

    return mean_fps, p95_fps, jitter




# =====================================================
# RealMax 核心：1:1 分辨率构造
# =====================================================
def make_1_1(hw, multiple=1):
    side = align_down(hw, multiple)
    return side, side



# =====================================================
# RealMax Anchor + 二分 搜索
# =====================================================
def realmax_search_MSRR(net, device, fps_sample, log_file,
                        target_fps=30.0,
                        tol_H=8,
                        coarse_loops=20,
                        precision_loops=100,
                        precision_warmup=20,
                        verify_margin=4,
                        size_multiple=1):
    log(log_file, "\n================ RealMax MSRR Search (Coarse+Precision) ================\n")


    last_pass = None
    first_fail = None

    # --- Anchor phase (Coarse) ---
    for H in anchors:
        W, H2 = make_1_1(H, multiple=size_multiple)
        img = prepare_fps_input(fps_sample, device, (W, H2))
        fps_c, _, oom = safe_measure(net, img, loops=coarse_loops, warmup=10, return_jitter=False)

        if oom:
            log(log_file, f"[OOM][Coarse] Anchor H={H2} ({W}x{H2}) -> FAIL")
            first_fail = H2
            break

        if fps_c >= target_fps:
            log(log_file, f"[PASS][Coarse] Anchor H={H2} ({W}x{H2})  FPS={fps_c:.2f}")
            last_pass = H2
        else:
            log(log_file, f"[FAIL][Coarse] Anchor H={H2} ({W}x{H2})  FPS={fps_c:.2f}")
            first_fail = H2
            break

    if last_pass is None:
        log(log_file, "❌ Even smallest resolution failed. MSRR = -1")
        return -1.0

    if first_fail is None:
        log(log_file, "✔ All anchors passed. Assume >= 2160p")
        first_fail = anchors[-1] * 2

    # --- Binary Search (Precision) ---
    low = last_pass
    high = first_fail

    m = max(1, int(size_multiple))
    low_k = low // m
    high_k = high // m
    tol_k = max(1, int(math.ceil(tol_H / m)))

    confirmed_pass = low
    round_id = 1

    while (high_k - low_k) > tol_k:
        mid_k = (low_k + high_k) // 2
        Hm = mid_k * m
        Wm, Hm = make_1_1(Hm, multiple=m)

        img = prepare_fps_input(fps_sample, device, (Wm, Hm))

        try:
            _, p95_fps, jit = measure_precision_p95(net, img, loops=precision_loops, warmup=precision_warmup)
        except RuntimeError as e:
            if is_oom_error(e):
                cleanup_cuda("OOM")
                log(log_file, f"[Round {round_id}] Try H={Hm} -> Precision OOM => FAIL")
                high_k = mid_k
                round_id += 1
                continue
            raise

        if p95_fps >= target_fps:
            log(log_file, f"[Round {round_id}] Try H={Hm} P95FPS={p95_fps:.2f} -> PASS")
            low_k = mid_k
            confirmed_pass = Hm
        else:
            log(log_file, f"[Round {round_id}] Try H={Hm} P95FPS={p95_fps:.2f} -> FAIL")
            high_k = mid_k

        eps = 0.01 * target_fps
        if abs(p95_fps - target_fps) <= eps:
            break
        round_id += 1

    # ==========================================================
    # Finalize (NEW): 先精测 candidate；若击穿则按 anchors 回滚精测，
    # 找到 PASS anchor 后在 (PASS, FAIL) 区间做精测二分收敛。
    # ==========================================================
    min_anchor_h = make_1_1(anchors[0], multiple=size_multiple)[1]

    def precision_p95_at(h):
        Wt, Ht = make_1_1(h, multiple=size_multiple)
        img_t = prepare_fps_input(fps_sample, device, (Wt, Ht))
        try:
            _, p95_fps, jit = measure_precision_p95(net, img_t, loops=precision_loops, warmup=precision_warmup)
            return (p95_fps >= target_fps), p95_fps, jit, False
        except RuntimeError as e:
            if is_oom_error(e):
                cleanup_cuda("OOM")
                return False, None, None, True
            raise

    # 把 anchors 对齐到 multiple 后去重排序，作为“回滚阶梯”
    anchors_aligned = sorted({make_1_1(a, multiple=size_multiple)[1] for a in anchors})
    min_anchor_h = anchors_aligned[0]

    # candidate：精测二分阶段得到的“最大 PASS 点”，但可能会在精测下击穿
    candidate = confirmed_pass if (confirmed_pass is not None) else low

    # 1) 先精测 candidate
    ok_cand, p95_cand, jit_cand, oom_cand = precision_p95_at(candidate)

    if (not oom_cand) and ok_cand:
        # candidate 精测仍 PASS：直接选它
        msrr_h = candidate
        final_fps = p95_cand
        final_jitter = jit_cand
        log(log_file, f"[Finalize] H={candidate} -> PASS P95FPS={p95_cand:.2f} (SELECTED)")
    else:
        # 2) candidate 精测击穿 / OOM：改为 anchor 回滚 + 精测二分
        if oom_cand:
            log(log_file, f"[Finalize] H={candidate} -> OOM => treat as FAIL, start anchor rollback")
        else:
            log(log_file, f"[Finalize] H={candidate} -> FAIL P95FPS={p95_cand:.2f}, start anchor rollback")

        # high_h：已知 FAIL 的点（上界）
        high_h = candidate

        # 找 candidate 下方最近的 anchor 索引（例如 candidate=1000~1080 => 最近下方 anchor=900）
        idx = -1
        for i, a in enumerate(anchors_aligned):
            if a < candidate:
                idx = i
            else:
                break
        if idx < 0:
            # candidate 已经 <= 最小 anchor（理论上不该发生，但兜底）
            idx = 0

        found_low = None
        found_low_fps = None
        found_low_jit = None

        # 2.1) 从最近下方 anchor 开始向下回滚精测，直到找到 PASS
        for j in range(idx, -1, -1):
            ah = anchors_aligned[j]

            ok_a, p95_a, jit_a, oom_a = precision_p95_at(ah)
            if oom_a:
                log(log_file, f"[Finalize-RB] Anchor H={ah} -> OOM => FAIL")
                high_h = ah
                continue

            if ok_a:
                log(log_file, f"[Finalize-RB] Anchor H={ah} -> PASS P95FPS={p95_a:.2f} (LOW FOUND)")
                found_low = ah
                found_low_fps = p95_a
                found_low_jit = jit_a
                break
            else:
                log(log_file, f"[Finalize-RB] Anchor H={ah} -> FAIL P95FPS={p95_a:.2f}")
                high_h = ah  # FAIL 点继续下移，缩小上界

        if found_low is None:
            # 连最小 anchor 精测都不过
            msrr_h = -1
            mp_val = -1.0
            log(log_file, "[WARN] Anchor rollback reached min anchor and still FAIL. MSRR = -1.")
            log(log_file, "\n------ RealMax Result ------")
            log(log_file, f"MSRR Height = {msrr_h}")
            log(log_file, f"Megapixels  = {mp_val:.3f} MP")
            log(log_file, "----------------------------\n")
            return mp_val

        # 3) 在 [found_low, high_h) 区间做“精测二分”，找最大 PASS
        low_h = found_low

        low_k2 = low_h // m
        high_k2 = max(low_k2 + 1, high_h // m)  # 确保 high_k2 > low_k2
        tol_k2 = max(1, int(math.ceil(tol_H / m)))

        confirmed_h = low_h
        final_fps = found_low_fps
        final_jitter = found_low_jit

        round_id2 = 1
        while (high_k2 - low_k2) > tol_k2:
            mid_k2 = (low_k2 + high_k2) // 2
            Hm2 = mid_k2 * m
            Wm2, Hm2 = make_1_1(Hm2, multiple=m)

            ok_m, p95_m, jit_m, oom_m = precision_p95_at(Hm2)

            if oom_m:
                log(log_file, f"[Finalize-Bin {round_id2}] Try H={Hm2} -> OOM => FAIL")
                high_k2 = mid_k2
            elif ok_m:
                log(log_file, f"[Finalize-Bin {round_id2}] Try H={Hm2} P95FPS={p95_m:.2f} -> PASS")
                low_k2 = mid_k2
                confirmed_h = Hm2
                final_fps = p95_m
                final_jitter = jit_m
            else:
                log(log_file, f"[Finalize-Bin {round_id2}] Try H={Hm2} P95FPS={p95_m:.2f} -> FAIL")
                high_k2 = mid_k2

            round_id2 += 1

        msrr_h = confirmed_h
        log(log_file, f"[Finalize] Precision-binary selected H={msrr_h} P95FPS={final_fps:.2f}")

    # --- Result ---
    if msrr_h < 0:
        mp_val = -1.0
    else:
        Wf, Hf = make_1_1(msrr_h, multiple=size_multiple)
        mp_val = (Wf * Hf) / 1e6

    log(log_file, "\n------ RealMax Result ------")
    log(log_file, f"MSRR Height = {msrr_h}")
    log(log_file, f"Megapixels  = {mp_val:.3f} MP")
    log(log_file, "----------------------------\n")

    return mp_val

# =====================================================
# 参数量
# =====================================================
def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


# =====================================================
#  新增：API 封装类与函数
# =====================================================
class ApiLogger:
    """用于 API 调用的伪文件对象"""

    def __init__(self, debug=False):
        self.debug = debug
        self.is_api_logger = True

    def write(self, s):
        pass

    def flush(self):
        pass



# =====================================================
# ==================== 外部调用 API ====================
# =====================================================
def msrr(model, size_multiple=1, debug=False):
    """
    外部调用 API。
    Return:
        float: MSRR Megapixels
    """
    # 1. 检测设备
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 确保模型在 Eval 模式
    model.eval()

    # 3. 构造 Logger
    logger = ApiLogger(debug=debug)

    # 4. 全局预热 (传入 None 生成虚拟图片)
    if debug:
        print("Starting Global Warmup...")

    # 注意：API模式下为了快，min_frames可以设小一点，或者保持原样
    global_warmup(model, device, fps_sample=None, size_wh=(256, 256), min_frames=20, multiple=size_multiple)

    # 5. 执行搜索
    mp = realmax_search_MSRR(
        net=model,
        device=device,
        fps_sample=None,
        log_file=logger,
        target_fps=30.0,
        tol_H=8,
        coarse_loops=20,
        precision_loops=50,  # API模式可适当降低轮数
        precision_warmup=10,
        verify_margin=4,
        size_multiple=size_multiple
    )

    return mp

def score(model, size_multiple=1, debug=False):
    """
    RealMax Score API (P95-based).
    Computes MSRR@{60,30,24} (in MP) under P95 real-time constraint, then returns a weighted score.

    Default weights:
        60 FPS -> 0.2
        30 FPS -> 0.4
        24 FPS -> 0.4

    Args:
        model: torch.nn.Module
        size_multiple (int): force H/W to be multiple of N (e.g., 8 or 32)
        debug (bool): print debug logs

    Returns:
        float: weighted score in Megapixels (MP-weighted).
    """
    # ------------------------------
    # 0) Setup device / eval / logger
    # ------------------------------
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    logger = ApiLogger(debug=debug)

    m = max(1, int(size_multiple))

    # ------------------------------
    # 1) Global warmup (same as msrr)
    # ------------------------------
    if debug:
        print("Starting Global Warmup (Score)...")
    global_warmup(model, device, fps_sample=None, size_wh=(256, 256), min_frames=20, multiple=m)

    # ------------------------------
    # 2) One-time coarse sweep to bracket {60,30,24}
    # ------------------------------
    log(logger, "\n================ RealMax Score (Coarse Bracketing) ================\n")

    anchors_aligned = sorted({make_1_1(a, multiple=m)[1] for a in anchors})

    fps_targets = [60.0, 30.0, 24.0]  # search order will be 60 -> 30 -> 24
    brackets = {f: {"last_pass": None, "first_fail": None} for f in fps_targets}

    coarse_loops = 20
    coarse_warmup = 10

    for H in anchors_aligned:
        W, H2 = make_1_1(H, multiple=m)
        img = prepare_fps_input(None, device, (W, H2))

        fps_c, _, oom = safe_measure(model, img, loops=coarse_loops, warmup=coarse_warmup, return_jitter=False)

        if oom:
            log(logger, f"[OOM][Coarse] Anchor H={H2} ({W}x{H2}) -> FAIL for all targets")
            # OOM at this resolution => treat as first_fail for any target not yet failed
            for f in fps_targets:
                if brackets[f]["first_fail"] is None:
                    brackets[f]["first_fail"] = H2
            break

        log(logger, f"[Coarse] Anchor H={H2} ({W}x{H2})  meanFPS={fps_c:.2f}")

        # update each bracket independently
        for f in fps_targets:
            if brackets[f]["first_fail"] is not None:
                continue  # already found fail boundary

            if fps_c >= f:
                brackets[f]["last_pass"] = H2
            else:
                brackets[f]["first_fail"] = H2

        # If all three already got first_fail, we can stop early
        if all(brackets[f]["first_fail"] is not None for f in fps_targets):
            break

    max_anchor = anchors_aligned[-1]

    # fill missing first_fail if never failed in coarse sweep
    for f in fps_targets:
        if brackets[f]["first_fail"] is None:
            brackets[f]["first_fail"] = max_anchor * 2
            log(logger, f"[Coarse] Target {f:.0f}FPS: all anchors passed, set first_fail={brackets[f]['first_fail']}")

        if brackets[f]["last_pass"] is None:
            log(logger, f"[Coarse] Target {f:.0f}FPS: even smallest anchor failed -> MSRR@{f:.0f} = 0 MP")

    # ------------------------------
    # 3) Precision search WITH bounds (binary + finalize rollback)
    #    We reuse your MSRR finalize logic but skip anchor phase.
    # ------------------------------
    def search_msrr_mp_with_bounds(target_fps, low_h, high_h,
                                   tol_H=8,
                                   precision_loops=50,
                                   precision_warmup=10):
        """
        Returns MSRR in MP for a given target_fps using:
        - precision binary in [low_h, high_h)
        - finalize: re-test candidate; if fail/OOM => anchor rollback + precision binary
        """
        if low_h is None:
            return 0.0

        # ---- Binary Search (Precision) ----
        low = int(low_h)
        high = int(high_h)

        low_k = low // m
        high_k = max(low_k + 1, high // m)
        tol_k = max(1, int(math.ceil(tol_H / m)))

        confirmed_pass = low
        round_id = 1

        log(logger, f"\n================ Score Precision Search @ {target_fps:.0f}FPS ================\n")
        log(logger, f"[Init] low={low} high={high} (multiple={m})")

        while (high_k - low_k) > tol_k:
            mid_k = (low_k + high_k) // 2
            Hm = mid_k * m
            Wm, Hm = make_1_1(Hm, multiple=m)
            img = prepare_fps_input(None, device, (Wm, Hm))

            try:
                _, p95_fps, jit = measure_precision_p95(model, img, loops=precision_loops, warmup=precision_warmup)
            except RuntimeError as e:
                if is_oom_error(e):
                    cleanup_cuda("OOM")
                    log(logger, f"[Round {round_id}] Try H={Hm} -> Precision OOM => FAIL")
                    high_k = mid_k
                    round_id += 1
                    continue
                raise

            if p95_fps >= target_fps:
                log(logger, f"[Round {round_id}] Try H={Hm} P95FPS={p95_fps:.2f} -> PASS")
                low_k = mid_k
                confirmed_pass = Hm
            else:
                log(logger, f"[Round {round_id}] Try H={Hm} P95FPS={p95_fps:.2f} -> FAIL")
                high_k = mid_k

            eps = 0.01 * target_fps
            if abs(p95_fps - target_fps) <= eps:
                break
            round_id += 1

        # ---- Finalize (same spirit as your MSRR) ----
        def precision_p95_at(h):
            Wt, Ht = make_1_1(h, multiple=m)
            img_t = prepare_fps_input(None, device, (Wt, Ht))
            try:
                _, p95_fps, jit = measure_precision_p95(model, img_t, loops=precision_loops, warmup=precision_warmup)
                return (p95_fps >= target_fps), p95_fps, jit, False
            except RuntimeError as e:
                if is_oom_error(e):
                    cleanup_cuda("OOM")
                    return False, None, None, True
                raise

        candidate = confirmed_pass if confirmed_pass is not None else low
        ok_cand, p95_cand, jit_cand, oom_cand = precision_p95_at(candidate)

        if (not oom_cand) and ok_cand:
            msrr_h = candidate
            log(logger, f"[Finalize] H={candidate} -> PASS P95FPS={p95_cand:.2f} (SELECTED)")
        else:
            if oom_cand:
                log(logger, f"[Finalize] H={candidate} -> OOM => treat as FAIL, start anchor rollback")
            else:
                log(logger, f"[Finalize] H={candidate} -> FAIL P95FPS={p95_cand:.2f}, start anchor rollback")

            high_h2 = candidate

            # find nearest aligned anchor below candidate
            idx = -1
            for i, a in enumerate(anchors_aligned):
                if a < candidate:
                    idx = i
                else:
                    break
            if idx < 0:
                idx = 0

            found_low = None

            # rollback downward until find PASS
            for j in range(idx, -1, -1):
                ah = anchors_aligned[j]
                ok_a, p95_a, jit_a, oom_a = precision_p95_at(ah)

                if oom_a:
                    log(logger, f"[Finalize-RB] Anchor H={ah} -> OOM => FAIL")
                    high_h2 = ah
                    continue

                if ok_a:
                    log(logger, f"[Finalize-RB] Anchor H={ah} -> PASS P95FPS={p95_a:.2f} (LOW FOUND)")
                    found_low = ah
                    break
                else:
                    log(logger, f"[Finalize-RB] Anchor H={ah} -> FAIL P95FPS={p95_a:.2f}")
                    high_h2 = ah

            if found_low is None:
                log(logger, f"[WARN] Rollback reached min anchor and still FAIL for {target_fps:.0f}FPS. Return 0 MP.")
                return 0.0

            # precision binary within [found_low, high_h2)
            low_h2 = found_low
            low_k2 = low_h2 // m
            high_k2 = max(low_k2 + 1, high_h2 // m)
            tol_k2 = max(1, int(math.ceil(tol_H / m)))

            confirmed_h = low_h2
            round_id2 = 1

            while (high_k2 - low_k2) > tol_k2:
                mid_k2 = (low_k2 + high_k2) // 2
                Hm2 = mid_k2 * m
                Wm2, Hm2 = make_1_1(Hm2, multiple=m)

                ok_m, p95_m, jit_m, oom_m = precision_p95_at(Hm2)

                if oom_m:
                    log(logger, f"[Finalize-Bin {round_id2}] Try H={Hm2} -> OOM => FAIL")
                    high_k2 = mid_k2
                elif ok_m:
                    log(logger, f"[Finalize-Bin {round_id2}] Try H={Hm2} P95FPS={p95_m:.2f} -> PASS")
                    low_k2 = mid_k2
                    confirmed_h = Hm2
                else:
                    log(logger, f"[Finalize-Bin {round_id2}] Try H={Hm2} P95FPS={p95_m:.2f} -> FAIL")
                    high_k2 = mid_k2

                round_id2 += 1

            msrr_h = confirmed_h
            log(logger, f"[Finalize] Precision-binary selected H={msrr_h}")

        Wf, Hf = make_1_1(msrr_h, multiple=m)
        mp_val = (Wf * Hf) / 1e6
        log(logger, f"[Result @ {target_fps:.0f}FPS] MSRR Height={msrr_h}, MP={mp_val:.3f}")
        return mp_val

    # ------------------------------
    # 4) Nested precision searches: 60 -> 30 -> 24
    # ------------------------------
    # weights: 60/30/24 -> 0.2/0.4/0.4
    w60, w30, w24 = 0.2, 0.4, 0.4

    # Brackets from coarse
    low60, high60 = brackets[60.0]["last_pass"], brackets[60.0]["first_fail"]
    mp60 = search_msrr_mp_with_bounds(60.0, low60, high60)

    # For 30FPS, low can start at height corresponding to mp60's found pass (monotonicity),
    # but we only have MP. We'll reuse coarse low30; if mp60>0, we can safely set low30 >= low60.
    low30, high30 = brackets[30.0]["last_pass"], brackets[30.0]["first_fail"]
    if (low30 is not None) and (low60 is not None):
        low30 = max(int(low30), int(low60))
    mp30 = search_msrr_mp_with_bounds(30.0, low30, high30)

    low24, high24 = brackets[24.0]["last_pass"], brackets[24.0]["first_fail"]
    if (low24 is not None) and (low30 is not None):
        low24 = max(int(low24), int(low30))
    mp24 = search_msrr_mp_with_bounds(24.0, low24, high24)

    # Safety: if any returned negative (shouldn't here), clamp to 0
    mp60 = 0.0 if (mp60 is None or mp60 < 0) else float(mp60)
    mp30 = 0.0 if (mp30 is None or mp30 < 0) else float(mp30)
    mp24 = 0.0 if (mp24 is None or mp24 < 0) else float(mp24)

    score_mp = (w60 * mp60) + (w30 * mp30) + (w24 * mp24)

    log(logger, "\n================ RealMax Score Result ================\n")
    log(logger, f"MSRR@60FPS: {mp60:.3f} MP (w={w60})")
    log(logger, f"MSRR@30FPS: {mp30:.3f} MP (w={w30})")
    log(logger, f"MSRR@24FPS: {mp24:.3f} MP (w={w24})")
    log(logger, f"RealMax Score (weighted MP): {score_mp:.3f}")
    log(logger, "======================================================\n")

    return score_mp

def ave_fps(model, size_multiple=1, debug=False):
    """
    Average FPS sweep API (with precision-style loops/warmup).
    Usage:
        results = realmax.ave_fps(net, size_multiple=32, debug=True)

    Protocol:
      - global warmup once
      - evaluate a fixed set of aligned anchor resolutions from low->high
      - each point uses precision measurement loops/warmup (loops=100, warmup=20)
      - stop early if mean_fps < 5 or OOM

    Returns:
      List[Dict]: [
        {"H": int, "W": int, "MP": float, "mean_fps": float, "p95_fps": float, "jitter": float, "status": "OK"/"STOP"/"OOM"}
      ]
    """
    # 1) device / eval / logger
    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    logger = ApiLogger(debug=debug)

    m = max(1, int(size_multiple))

    # 2) global warmup (same philosophy as msrr/score)
    if debug:
        print("Starting Global Warmup (AveFPS)...")
    global_warmup(model, device, fps_sample=None, size_wh=(256, 256), min_frames=20, multiple=m)

    # 3) fixed anchor points (same as MSRR), aligned + de-duplicated
    anchors_aligned = sorted({make_1_1(a, multiple=m)[1] for a in anchors})

    # precision-style settings (match measure_precision_p95 default behavior)
    precision_loops = 100
    precision_warmup = 20

    stop_fps_threshold = 5.0

    log(logger, "\n================ RealMax AveFPS Sweep (Precision loops) ================\n")
    log(logger, f"[Config] multiple={m}, loops={precision_loops}, warmup={precision_warmup}, stop_if_mean_fps<{stop_fps_threshold}\n")

    results = []

    for H in anchors_aligned:
        W, H2 = make_1_1(H, multiple=m)
        img = prepare_fps_input(None, device, (W, H2))

        # precision measurement returns (mean_fps, p95_fps, jitter)
        try:
            mean_fps, p95_fps, jit = measure_precision_p95(model, img, loops=precision_loops, warmup=precision_warmup)
        except RuntimeError as e:
            if is_oom_error(e):
                cleanup_cuda("OOM")
                mp_val = (W * H2) / 1e6
                log(logger, f"[OOM] H={H2} ({W}x{H2}, {mp_val:.3f}MP) -> STOP")
                results.append({
                    "H": int(H2), "W": int(W), "MP": float(mp_val),
                    "mean_fps": None, "p95_fps": None, "jitter": None,
                    "status": "OOM"
                })
                break
            raise

        mp_val = (W * H2) / 1e6
        log(logger, f"[OK] H={H2} ({W}x{H2}, {mp_val:.3f}MP) meanFPS={mean_fps:.2f}  P95FPS={p95_fps:.2f}  jitter%={jit:.2f}")

        results.append({
            "H": int(H2), "W": int(W), "MP": float(mp_val),
            "mean_fps": float(mean_fps), "p95_fps": float(p95_fps), "jitter": float(jit),
            "status": "OK"
        })

        # early stop condition
        if mean_fps < stop_fps_threshold:
            log(logger, f"[STOP] meanFPS {mean_fps:.2f} < {stop_fps_threshold} at H={H2}, stop further testing.")
            # mark the last entry as STOP (optional; keeps it explicit)
            results[-1]["status"] = "STOP"
            break

    log(logger, "\n================ AveFPS Sweep Done ================\n")
    return results

