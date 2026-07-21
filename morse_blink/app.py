"""Real-time Morse blink recognition application with OpenCV UI."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from morse_blink.detector import BlinkDetector, BlinkPhase
from morse_blink.morse import MorseDecoder


def _draw_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int],
    line_height: int = 28,
) -> None:
    x, y = origin
    for i, (text, color) in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_status_bar(frame: np.ndarray, ear: float, phase: BlinkPhase, threshold: float) -> None:
    height, width = frame.shape[:2]
    bar_width = int(min(width * 0.4, 200))
    fill = int(np.clip(ear / 0.35, 0.0, 1.0) * bar_width)
    x0, y0 = 10, height - 30

    cv2.rectangle(frame, (x0, y0), (x0 + bar_width, y0 + 16), (60, 60, 60), -1)
    color = (0, 0, 255) if phase == BlinkPhase.CLOSED else (0, 200, 0)
    cv2.rectangle(frame, (x0, y0), (x0 + fill, y0 + 16), color, -1)

    thresh_x = x0 + int(np.clip(threshold / 0.35, 0.0, 1.0) * bar_width)
    cv2.line(frame, (thresh_x, y0 - 2), (thresh_x, y0 + 18), (255, 255, 0), 2)
    cv2.putText(
        frame,
        f"EAR {ear:.2f}",
        (x0 + bar_width + 10, y0 + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run(
    camera_index: int = 0,
    ear_threshold: float = 0.21,
    dot_max_duration: float = 0.35,
    letter_gap: float = 1.2,
    word_gap: float = 2.5,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open webcam. Check camera permissions and index.", file=sys.stderr)
        sys.exit(1)

    detector = BlinkDetector(
        ear_threshold=ear_threshold,
        dot_max_duration=dot_max_duration,
    )
    decoder = MorseDecoder(letter_gap=letter_gap, word_gap=word_gap)

    last_blink_label = ""
    last_blink_time = 0.0
    fps_time = time.monotonic()
    fps = 0.0

    print("Morse Blink started. Controls: Q=quit, R=reset, C=calibration hint")
    print("  Short blink (<0.35s) = dot (.)  |  Long blink = dash (-)")
    print("  Pause ~1.2s = next letter       |  Pause ~2.5s = space")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            result = detector.process(frame)

            now = time.monotonic()
            frame_delta = now - fps_time
            if frame_delta > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / frame_delta)
            fps_time = now

            if result.blink is not None:
                symbol = "." if result.blink.is_dot else "-"
                decoder.add_symbol(symbol)
                kind = "DOT" if result.blink.is_dot else "DASH"
                last_blink_label = f"{kind} ({result.blink.duration:.2f}s)"
                last_blink_time = now

            decoder.tick()

            if result.face_detected:
                _draw_status_bar(frame, result.ear, result.phase, ear_threshold)
            else:
                cv2.putText(
                    frame,
                    "No face detected",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            overlay_lines = [
                ("Morse Blink Decoder", (0, 255, 180)),
                (f"FPS: {fps:.0f}", (200, 200, 200)),
                ("", (255, 255, 255)),
                (f"Pattern: {decoder.current_pattern or '-'}", (255, 220, 100)),
                (f"Message: {decoder.decoded_text or '-'}", (255, 255, 255)),
            ]

            if now - last_blink_time < 1.5 and last_blink_label:
                overlay_lines.append((f"Last blink: {last_blink_label}", (100, 255, 100)))

            overlay_lines.extend([
                ("", (255, 255, 255)),
                ("Q quit | R reset", (160, 160, 160)),
            ])

            _draw_panel(frame, overlay_lines, (10, 30))

            cv2.imshow("Morse Blink — Eye Blink Morse Code", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                decoder.reset()
                last_blink_label = ""
            if key in (ord("c"), ord("C")):
                print(
                    f"\n[Calibration] Current EAR={result.ear:.3f}, "
                    f"threshold={ear_threshold:.3f}, phase={result.phase.name}"
                )

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

        if decoder.decoded_text:
            print(f"\nFinal message: {decoder.decoded_text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode Morse code from eye blinks using your webcam.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument(
        "--ear-threshold",
        type=float,
        default=0.21,
        help="EAR below this value counts as eyes closed (default: 0.21)",
    )
    parser.add_argument(
        "--dot-max",
        type=float,
        default=0.35,
        help="Blink duration at or below this is a dot (default: 0.35s)",
    )
    parser.add_argument(
        "--letter-gap",
        type=float,
        default=1.2,
        help="Seconds of idle time to finish a letter (default: 1.2)",
    )
    parser.add_argument(
        "--word-gap",
        type=float,
        default=2.5,
        help="Seconds of idle time to finish a word/space (default: 2.5)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(
        camera_index=args.camera,
        ear_threshold=args.ear_threshold,
        dot_max_duration=args.dot_max,
        letter_gap=args.letter_gap,
        word_gap=args.word_gap,
    )


if __name__ == "__main__":
    main()
