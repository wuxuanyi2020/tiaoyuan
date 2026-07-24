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


def build_parser():
    parser = argparse.ArgumentParser(description="立定跳远自动检测系统")
    parser.add_argument("--video", type=str, default="0", help="视频文件路径或摄像头索引 (默认 0)")
    parser.add_argument("--save", type=str, default="result.json", help="结果 JSON 文件名")
    parser.add_argument("--no-display", action="store_true", help="不显示预览窗口（批量模式默认启用）")
    parser.add_argument("--backend", type=str, default="mediapipe")
    parser.add_argument("--debug-dir", type=str, default=None)
    parser.add_argument("--record", type=str, default=None, help="录制输出视频路径（旧参数，需要手动给路径）")
    parser.add_argument("--output-video", nargs="?", const="auto", default=None, metavar="PATH",
                        help="输出运行时程序画面视频；不写 PATH 时自动保存到本次结果目录 run_view.mp4")
    parser.add_argument("--mat-length-cm", type=float, default=338.0)
    parser.add_argument("--mat-width-cm", type=float, default=100.0)
    parser.add_argument("--trigger-move-cm", type=float, default=32.0)
    parser.add_argument("--trigger-frames", type=int, default=2)
    parser.add_argument("--takeoff-backtrack-frames", type=int, default=2,
                        help="起跳触发帧倒推多少帧作为正式起跳帧，默认2帧")
    parser.add_argument("--min-flight-frames", type=int, default=10,
                        help="起跳触发帧到最早有效落地帧的最小间隔，默认10帧，避免半空误判落地")
    parser.add_argument("--max-jump-frames", type=int, default=30,
                        help="起跳后最长等待落地帧数，默认30帧(30fps约1秒)，超时仍无可靠触地则不给成绩")
    parser.add_argument("--takeoff-line-cm", type=float, default=31.0,
                        help="起跳线距起跳端的距离(cm)，默认31；--jump-direction rtl 时会自动换算到垫子右侧")
    parser.add_argument("--jump-direction", choices=["ltr", "rtl"], default="ltr",
                        help="跳跃方向：ltr=画面左到右(默认)，rtl=画面右到左/起跳线在右侧")
    parser.add_argument("--takeoff-offset-cm", type=float, default=3.0)
    parser.add_argument("--manual-calib", action="store_true", help="手动四点标定（需鼠标点击）")
    parser.add_argument("--no-foul-detection", action="store_true", help="禁用犯规检测（默认开启）")
    parser.add_argument("--landing-offset-cm", type=float, default=-5.0,
                        help="落地点修正值(cm)，补偿鞋后跟厚度，默认-5.0（负值缩短距离）")
    parser.add_argument("--debug", action="store_true", help="调试模式：输出起跳/落地触发条件到日志")
    parser.add_argument("--diff", action="store_true", help="启用 MOG2 背景差分法距离修正（默认关闭）")
    parser.add_argument("--yolo", nargs=2, metavar=("VERSION", "SCALE"),
                        help="启用 YOLO 实例分割距离修正，指定版本和尺度，如 --yolo 26 x（版本: 8/11/26, 尺度: n/s/m/l/x）")
    parser.add_argument("--enable-mat-output", action="store_true", help="输出垫子识别图 (mat_mask_quad/hsv)")
    parser.add_argument("--test-grid", action="store_true", help="输出垫子毫米格测试图")
    parser.add_argument("--stream-mode", action="store_true", help="模拟流模式：完成一跳后不中断，垫内无人时重新标定并继续输出多跳成绩")
    parser.add_argument("--stream-recalib-empty-frames", type=int, default=15, help="流模式下垫内无人连续多少帧后重新标定并锁定，默认15帧")
    parser.add_argument("--ocr-time", action="store_true", help="识别左上角模拟时间码，并在结果中输出到帧的 timecode")
    parser.add_argument("--ocr-roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), default=(15, 18, 155, 30), help="左上角时间 OCR 区域，默认 15 18 155 30")
    # 批量模式
    parser.add_argument("--batch", action="store_true", help="批量处理 videos/ 下所有视频（跳远1-1 ~ 跳远1-9）")
    parser.add_argument("--videos", nargs="*", default=None, help="批量处理指定的视频列表")
    return parser



def resolve_record_path(args, result_dir, video_name):
    """解析运行画面录制路径。

    --record 保持兼容：用户必须给完整路径；
    --output-video 是新参数：单独使用时自动写到本次结果目录。
    """
    if getattr(args, "record", None):
        return str(args.record).strip().strip('"').strip("'")
    output_video = getattr(args, "output_video", None)
    if output_video is None:
        return None
    if output_video == "auto":
        return os.path.join(result_dir, "run_view.mp4")
    path = str(output_video).strip().strip('"').strip("'")
    if not path:
        return os.path.join(result_dir, "run_view.mp4")
    root, ext = os.path.splitext(path)
    if os.path.isdir(path) or not ext:
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, f"{video_name}_run_view.mp4")
    return path

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
            display=False,
            backend=args.backend,
            debug_dir=args.debug_dir,
            record_path=resolve_record_path(args, result_dir, video_name),
            mat_length_cm=args.mat_length_cm,
            mat_width_cm=args.mat_width_cm,
            trigger_move_cm=args.trigger_move_cm,
            trigger_frames=args.trigger_frames,
            takeoff_backtrack_frames=args.takeoff_backtrack_frames,
            min_flight_frames=args.min_flight_frames,
            max_jump_frames=args.max_jump_frames,
            takeoff_line_cm=args.takeoff_line_cm,
            jump_direction=args.jump_direction,
            takeoff_offset_cm=args.takeoff_offset_cm,
            manual_calib=args.manual_calib,
            result_dir=result_dir,
            enable_foul_detection=not args.no_foul_detection,
            landing_offset_cm=args.landing_offset_cm,
            enable_diff=args.diff and not bool(args.yolo),
            enable_mat_output=args.enable_mat_output,
            enable_test_grid=args.test_grid,
            enable_seg=bool(args.yolo),
        yolo_version=args.yolo[0] if args.yolo else "11",
        yolo_scale=args.yolo[1] if args.yolo else "x",
        debug=args.debug,
        stream_mode=args.stream_mode,
        stream_recalib_empty_frames=args.stream_recalib_empty_frames,
        enable_ocr_time=args.ocr_time,
        ocr_roi=tuple(args.ocr_roi),
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
    yolo_time = result.get("yolo_infer_time_s", None)
    time_str = f" | yolo用时={yolo_time:.3f}s" if yolo_time is not None else ""
    print(f"<< 完成: {video_name} | 距离={dist_str} cm "
          f"| 有效={result.get('valid', 'N/A')} "
          f"| 犯规={result.get('foul_reason', '无')}"
          f"{time_str}")
    print("=" * 60)
    return result


def main():
    args = build_parser().parse_args()

    # 校验 --yolo 参数
    if args.yolo:
        ver, scale = args.yolo
        valid_vers = {"8", "11", "26"}
        valid_scales = {"n", "s", "m", "l", "x"}
        if ver not in valid_vers or scale not in valid_scales:
            print(f"错误: --yolo 版本必须是 {valid_vers}，尺度必须是 {valid_scales}，"
                  f"收到 '{ver} {scale}'")
            sys.exit(1)

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
        backend=args.backend,
        debug_dir=args.debug_dir,
        record_path=resolve_record_path(args, result_dir, video_name),
        mat_length_cm=args.mat_length_cm,
        mat_width_cm=args.mat_width_cm,
        trigger_move_cm=args.trigger_move_cm,
        trigger_frames=args.trigger_frames,
        takeoff_backtrack_frames=args.takeoff_backtrack_frames,
        min_flight_frames=args.min_flight_frames,
        max_jump_frames=args.max_jump_frames,
        takeoff_line_cm=args.takeoff_line_cm,
        jump_direction=args.jump_direction,
        takeoff_offset_cm=args.takeoff_offset_cm,
        manual_calib=args.manual_calib,
        result_dir=result_dir,
        enable_foul_detection=not args.no_foul_detection,
        landing_offset_cm=args.landing_offset_cm,
        enable_diff=args.diff and not bool(args.yolo),
        enable_mat_output=args.enable_mat_output,
        enable_test_grid=args.test_grid,
        enable_seg=bool(args.yolo),
        yolo_version=args.yolo[0] if args.yolo else "11",
        yolo_scale=args.yolo[1] if args.yolo else "x",
        debug=args.debug,
        stream_mode=args.stream_mode,
        stream_recalib_empty_frames=args.stream_recalib_empty_frames,
        enable_ocr_time=args.ocr_time,
        ocr_roi=tuple(args.ocr_roi),
    )
    StandingLongJumpSystem(config).run()


if __name__ == "__main__":
    main()
