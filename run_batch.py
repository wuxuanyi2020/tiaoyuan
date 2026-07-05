"""批量运行脚本 — 依次处理 videos/ 下的所有视频。"""
import subprocess
import sys
import os
from datetime import datetime

CONDA_PYTHON = r"C:\Users\admin\.conda\envs\tiaoyuan\python.exe"
VIDEOS_DIR = "videos"
VIDEOS = [
    "跳远1-1.mp4",
    "跳远1-2.mp4",
    "跳远1-3.mp4",
    "跳远1-4.mp4",
    "跳远1-5（背景干扰）.mp4",
    "跳远1-6（单腿跳1）.mp4",
    "跳远1-7（单腿跳2）.mp4",
    "跳远1-8（过线跳）.mp4",
    "跳远1-9.mp4",
]


def main():
    results = {}
    summary_csv = os.path.join("result", f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs("result", exist_ok=True)

    for v in VIDEOS:
        path = os.path.join(VIDEOS_DIR, v)
        if not os.path.exists(path):
            print(f"[SKIP] {v} 不存在")
            continue

        print("=" * 60)
        print(f">> 开始处理: {v}")
        print("=" * 60)
        sys.stdout.flush()

        proc = subprocess.run(
            [CONDA_PYTHON, "main.py", "--video", path, "--no-display"],
            capture_output=True, text=True, timeout=600,
        )

        # 查找最后的结果目录
        result_dirs = [
            d for d in os.listdir("result")
            if d.startswith(os.path.splitext(v)[0]) and os.path.isdir(os.path.join("result", d))
        ]
        if result_dirs:
            latest = sorted(result_dirs)[-1]
            result_path = os.path.join("result", latest, "result.json")
            if os.path.exists(result_path):
                import json
                with open(result_path, "r", encoding="utf-8") as f:
                    r = json.load(f)
                results[v] = r
                dist = r.get("distance_cm", "")
                valid = r.get("valid", "")
                foul = r.get("foul_reason", "")
                print(f"<< {v}: 距离={dist} cm, 有效={valid}, 犯规={foul}")
            else:
                results[v] = None
                print(f"<< {v}: 无结果文件")
        else:
            results[v] = None
            print(f"<< {v}: 无结果目录")

        # 打印子进程的 stdout
        if proc.stdout.strip():
            for line in proc.stdout.strip().split("\n"):
                print(f"  [stdout] {line}")
        sys.stdout.flush()

    # 汇总
    print("\n" + "=" * 60)
    print("== 批量处理汇总 ==")
    print("=" * 60)
    with open(summary_csv, "w", encoding="utf-8-sig") as f:
        f.write("视频文件,距离(cm),有效,犯规原因\n")
        for name, r in results.items():
            if r:
                dist = r.get("distance_cm", "")
                valid = r.get("valid", "")
                foul = r.get("foul_reason", "")
                print(f"  {name:<25s}  距离={str(dist):>6s} cm  有效={'[OK]' if valid else '[X]'}  犯规={foul}")
                f.write(f"{name},{dist},{valid},{foul}\n")
            else:
                print(f"  {name:<25s}  [无结果]")
                f.write(f"{name},,,无结果\n")
    print("=" * 60)
    print(f">> 汇总 CSV: {summary_csv}")


if __name__ == "__main__":
    main()
