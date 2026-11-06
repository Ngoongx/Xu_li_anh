"""Command-line runner for a single image or all images in a folder."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from color_detector import open_config, process_image_path, save_result


def main() -> int:
    parser = argparse.ArgumentParser(description="Phát hiện vật thể theo màu trong ảnh")
    parser.add_argument("input", help="Đường dẫn ảnh hoặc thư mục ảnh")
    parser.add_argument("--output", default="results", help="Thư mục ghi ảnh kết quả và CSV")
    parser.add_argument("--config", default="config/colors.json", help="File JSON cấu hình HSV")
    args = parser.parse_args()

    source = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = open_config(args.config)
    paths = [source] if source.is_file() else sorted(
        p for p in source.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        and "tham_chieu" not in p.stem.lower())
    if not paths:
        parser.error(f"Không tìm thấy ảnh trong {source}")

    records = []
    for path in paths:
        try:
            result, stats = process_image_path(path, config)
        except (OSError, ValueError) as exc:
            print(f"LỖI {path}: {exc}")
            records.append({"file": path.name, "status": "ERROR", "error": str(exc)})
            continue
        output_path = out_dir / f"{path.stem}_result.png"
        save_result(output_path, result)
        row = {"file": path.name, "status": "OK", "total": stats["total"],
               **stats["counts"], "result": str(output_path)}
        records.append(row)
        print(json.dumps(row, ensure_ascii=False))

    csv_path = out_dir / "summary.csv"
    fieldnames = sorted({key for row in records for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Da ghi thong ke: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
