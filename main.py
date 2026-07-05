"""立定跳远自动检测系统 — 主入口 & 批量处理。

用法:
  # 单次运行（摄像头）
  python main.py

  # 批量处理 videos/ 下的所有视频
  python main.py --batch

  # 批量处理指定视频列表
  python main.py --videos 跳远1-1.mp4 跳远1-2.mp4
"""
import argparse
import os
import sys
from datetime import datetime

from src.config import JumpConfig, resolve_video_source, make_result_dir
from src.core.jump_system import StandingLongJumpSystem

# ── conda 环境中的 Python 执行器 ──
CONDA_PYTHON = r"C:\Users\admin\.conda\envs\tiaoyuan\python.exe"


def build_parser():
    parser = argparse.ArgumentParser(description="立定跳远自动检测系统")
    parser.add_argument("--video", type=str, default="0", help="视频文件路径或摄像头索引 (默认 0)")
    parser.add_argument("--save", type=str, default="result.json", help="结果 JSON 文件名")
    parser.add_argument("--no-display", action="store_true", help="不显示预览窗口（批量模式默认启用）")
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt")
    parser.add_argument("--backend", type=str, default="mediapipe")
    parser.add_argument("--debug-dir", type=str, default=None)
    parser.add_argument("--record", type=str, default=None, help="录制输出视频路径")
    parser.add_argument("--mat-length-cm", type=float, default=340.0)
    parser.add_argument("--mat-width-cm", type=float, default=90.0)
    parser.add_argument("--trigger-move-cm", type=float, default=30.0)
    parser.add_argument("--trigger-frames", type=int, default=2)
    parser.add_argument("--min-flight-frames", type=int, default=5)
    parser.add_argument("--max-jump-frames", type=int, default=120)
    parser.add_argument("--takeoff-line-cm", type=float, default=32.0)
    parser.add_argument("--takeoff-offset-cm", type=float, default=0.0)
    parser.add_argument("--manual-calib", action="store_true", help="手动四点标定（需鼠标点击）")
    # 批量模式
    parser.add_argument("--batch", action="store_true", help="批量处理 videos/ 下所有视频（跳远1-1 ~ 跳远1-9）")
    parser.add_argument("--videos", nargs="*", default=None, help="批量处理指定的视频列表")
    return parser


def run_single(video_path, args):
    """运行单个视频处理。"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print("\n" + "=" * 60)
    print(f">> 开始处理: {video_name}")
    print("=" * 60)

    # 创建带时间标签的结果文件夹
    result_dir = make_result_dir(video_name)
    save_path = os.path.join(result_dir, "result.json")

    config = JumpConfig(
        video_source=video_path,
        save_path=save_path,
        display=False,  # 批量模式下不显示窗口
        model=args.model,
        backend=args.backend,
        debug_dir=args.debug_dir,
        record_path=args.record,
        mat_length_cm=args.mat_length_cm,
        mat_width_cm=args.mat_width_cm,
        trigger_move_cm=args.trigger_move_cm,
        trigger_frames=args.trigger_frames,
        min_flight_frames=args.min_flight_frames,
        max_jump_frames=args.max_jump_frames,
        takeoff_line_cm=args.takeoff_line_cm,
        takeoff_offset_cm=args.takeoff_offset_cm,
        manual_calib=args.manual_calib,
        result_dir=result_dir,
    )
    StandingLongJumpSystem(config).run()

    # 读取结果
    result = {"distance_cm": None, "valid": None, "foul_reason": None}
    if os.path.exists(save_path):
        import json
        with open(save_path, "r", encoding="utf-8") as f:
            result = json.load(f)

    dist_val = result.get("distance_cm")
    dist_str = f"{dist_val:>6.1f}" if isinstance(dist_val, (int, float)) else "  N/A  "
    print(f"<< 完成: {video_name} | 距离={dist_str} cm "
          f"| 有效={result.get('valid', 'N/A')} "
          f"| 犯规={result.get('foul_reason', '无')}")
    print("=" * 60)
    return result


def main():
    args = build_parser().parse_args()

    # ── 批量模式 ──
    if args.batch or args.videos:
        if args.videos:
            # 用户指定了视频列表
            video_files = args.videos
        else:
            # 默认跑 跳远1-1 ~ 跳远1-9
            videos_dir = os.path.join(os.path.dirname(__file__) or ".", "videos")
            if not os.path.isdir(videos_dir):
                print(f"错误: videos 目录不存在 ({videos_dir})")
                sys.exit(1)
            video_files = [f"跳远1-{i}.mp4" for i in range(1, 10)]

        results = {}
        for vf in video_files:
            # 如果 vf 不是绝对路径，尝试拼接 videos/
            if not os.path.isabs(vf):
                path = os.path.join("videos", vf)
                if not os.path.exists(path):
                    print(f"[SKIP] 文件不存在 {path}")
                    continue
            else:
                path = vf
                if not os.path.exists(path):
                    print(f"[SKIP] 文件不存在 {path}")
                    continue

            result = run_single(path, args)
            results[os.path.basename(vf)] = result

        # ── 输出汇总 ──
        print("\n" + "=" * 60)
        print("== 批量处理汇总 ==")
        print("=" * 60)
        for name, r in results.items():
            dist = r.get("distance_cm", "N/A")
            if isinstance(dist, (int, float)):
                dist_str = f"{dist:>6.1f} cm"
            else:
                dist_str = "  N/A  "
            valid = "[OK]" if r.get("valid") else "[X]"
            foul = r.get("foul_reason") or "无"
            print(f"  {name:<20s}  {dist_str}  {valid}  犯规: {foul}")
        print("=" * 60)

        # 汇总写入 result/summary.csv
        os.makedirs("result", exist_ok=True)
        summary_path = os.path.join("result", f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(summary_path, "w", encoding="utf-8-sig") as f:
            f.write("视频文件,距离(cm),有效,犯规原因\n")
            for name, r in results.items():
                dist = r.get("distance_cm", "")
                valid = r.get("valid", "")
                foul = r.get("foul_reason", "")
                f.write(f"{name},{dist},{valid},{foul}\n")
        print(f">> 汇总 CSV: {summary_path}")
        return

    # ── 单次运行模式 ──
    video_name = os.path.splitext(os.path.basename(str(resolve_video_source(args.video))))[0]
    result_dir = make_result_dir(video_name)
    config = JumpConfig(
        video_source=resolve_video_source(args.video),
        display=not args.no_display,
        save_path=os.path.join(result_dir, "result.json"),
        model=args.model,
        backend=args.backend,
        debug_dir=args.debug_dir,
        record_path=args.record,
        mat_length_cm=args.mat_length_cm,
        mat_width_cm=args.mat_width_cm,
        trigger_move_cm=args.trigger_move_cm,
        trigger_frames=args.trigger_frames,
        min_flight_frames=args.min_flight_frames,
        max_jump_frames=args.max_jump_frames,
        takeoff_line_cm=args.takeoff_line_cm,
        takeoff_offset_cm=args.takeoff_offset_cm,
        manual_calib=args.manual_calib,
        result_dir=result_dir,
    )
    StandingLongJumpSystem(config).run()


if __name__ == "__main__":
    main()
