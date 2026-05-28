from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from app.inference import load_emotion_model, predict_emotion
from src.config import EMOTION_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time webcam emotion recognition.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_efficientnet_b2.pt")
    parser.add_argument("--model", default="efficientnet_b2")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--imagenet-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoothing", type=int, default=5, help="Number of recent frames to average.")
    parser.add_argument("--min-face-size", type=int, default=48)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def draw_label(frame, text: str, x: int, y: int) -> None:
    cv2.rectangle(frame, (x, y - 28), (x + max(230, 10 * len(text)), y + 5), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 6, y - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train EfficientNet-B2 first or pass --checkpoint."
        )

    model, device = load_emotion_model(
        checkpoint_path=checkpoint_path,
        model_name=args.model,
        device=args.device,
    )

    print(f"Loaded {args.model} from {checkpoint_path}")
    print(f"Device: {device}")
    print("Press 'q' to quit.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from: {cascade_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index: {args.camera}")

    prob_buffer = deque(maxlen=max(args.smoothing, 1))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read frame from camera.")
            break

        if args.mirror:
            frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(args.min_face_size, args.min_face_size),
        )

        face = largest_face(faces)

        if face is not None:
            x, y, w, h = face

            margin = int(0.15 * max(w, h))
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, frame.shape[1])
            y2 = min(y + h + margin, frame.shape[0])

            face_bgr = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            try:
                result = predict_emotion(
                    model=model,
                    device=device,
                    image=face_rgb,
                    image_size=args.image_size,
                    imagenet_norm=args.imagenet_norm,
                )

                prob_buffer.append(result["probability_array"])
                smoothed = np.mean(np.stack(prob_buffer, axis=0), axis=0)

                label_id = int(smoothed.argmax())
                label = EMOTION_LABELS[label_id]
                conf = float(smoothed[label_id])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 220, 70), 2)
                draw_label(frame, f"{label}: {conf:.1%}", x1, max(y1, 30))

            except Exception as exc:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                draw_label(frame, f"inference error: {exc}", 10, 30)
        else:
            prob_buffer.clear()
            draw_label(frame, "no face detected", 10, 30)

        cv2.imshow("Real-time Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
