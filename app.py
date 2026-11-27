import streamlit as st
import cv2
import pandas as pd
import tempfile
import time
from pathlib import Path
from ultralytics import YOLO

st.set_page_config(page_title="AI Smart Traffic Management", page_icon="🚦", layout="wide")
st.title("🚦 AI-Based Smart Traffic Management System")
st.caption("Vehicle detection • Classification • Tracking • Congestion analysis")

VEHICLE_NAMES = {2: "Car", 3: "Bike", 5: "Bus", 7: "Truck"}
DEFAULT_MODEL = "yolov8n.pt"

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

with st.sidebar:
    st.header("Configuration")
    model_path = st.text_input("YOLO model", value=DEFAULT_MODEL,
                               help="Default YOLOv8 nano weights are downloaded automatically on first run.")
    confidence = st.slider("Detection confidence", 0.10, 0.90, 0.25, 0.05)
    low_limit = st.number_input("Low traffic: maximum vehicles/frame", min_value=1, max_value=100, value=5)
    medium_limit = st.number_input("Moderate traffic: maximum vehicles/frame", min_value=2, max_value=200, value=15)
    frame_step = st.slider("Analyze every Nth frame", 1, 10, 2)
    st.info("Congestion is estimated from the number of detected vehicles in each analyzed frame. It is a project-level proxy, not a certified traffic measurement.")

uploaded = st.file_uploader("Upload a road/traffic video", type=["mp4", "avi", "mov", "mkv"])

def congestion_level(count, low, medium):
    if count <= low:
        return "Low"
    if count <= medium:
        return "Moderate"
    return "High"

if uploaded:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    st.video(input_path)
    if st.button("▶️ Analyze traffic video", type="primary"):
        try:
            model = load_model(model_path)
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Could not open this video. Try another MP4/AVI file.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_path = tempfile.mktemp(suffix=".mp4")
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            progress = st.progress(0)
            status = st.empty()
            preview = st.empty()
            rows = []
            unique_ids = set()
            class_totals = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}
            frame_number = 0
            analyzed = 0
            congestion_counts = {"Low": 0, "Moderate": 0, "High": 0}

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1

                if frame_number % frame_step == 0:
                    result = model.track(
                        frame, persist=True, tracker="bytetrack.yaml",
                        classes=list(VEHICLE_NAMES.keys()), conf=confidence, verbose=False
                    )[0]
                    boxes = result.boxes
                    current_count = 0
                    frame_class_counts = {"Car": 0, "Bike": 0, "Bus": 0, "Truck": 0}

                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            cls_id = int(box.cls[0].item())
                            label = VEHICLE_NAMES.get(cls_id, "Vehicle")
                            frame_class_counts[label] += 1
                            current_count += 1
                            track_id = int(box.id[0].item()) if box.id is not None else None
                            if track_id is not None:
                                unique_ids.add((label, track_id))
                            class_totals[label] += 1

                    level = congestion_level(current_count, low_limit, medium_limit)
                    congestion_counts[level] += 1
                    timestamp = frame_number / fps
                    rows.append({
                        "frame": frame_number, "time_seconds": round(timestamp, 2),
                        "vehicles_in_frame": current_count, "cars": frame_class_counts["Car"],
                        "bikes": frame_class_counts["Bike"], "buses": frame_class_counts["Bus"],
                        "trucks": frame_class_counts["Truck"], "congestion": level
                    })
                    annotated = result.plot()
                    cv2.putText(annotated, f"Vehicles: {current_count} | Congestion: {level}",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    frame = annotated
                    analyzed += 1
                    preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    status.write(f"Processed frame {frame_number} / {total_frames or '?'}")
                    if total_frames:
                        progress.progress(min(frame_number / total_frames, 1.0))

                writer.write(frame)

            cap.release()
            writer.release()
            progress.progress(1.0)

            if not rows:
                st.warning("No frames were analyzed. Reduce the frame step or check the video.")
                st.stop()

            df = pd.DataFrame(rows)
            avg_vehicles = df["vehicles_in_frame"].mean()
            overall = congestion_level(avg_vehicles, low_limit, medium_limit)

            st.subheader("Traffic analysis results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Unique tracked vehicles", len(unique_ids))
            c2.metric("Average vehicles/frame", f"{avg_vehicles:.1f}")
            c3.metric("Peak vehicles/frame", int(df["vehicles_in_frame"].max()))
            c4.metric("Overall congestion", overall)

            st.markdown("#### Vehicle detections")
            a, b, c, d = st.columns(4)
            a.metric("Car detections", class_totals["Car"])
            b.metric("Bike detections", class_totals["Bike"])
            c.metric("Bus detections", class_totals["Bus"])
            d.metric("Truck detections", class_totals["Truck"])

            st.markdown("#### Congestion over time")
            st.line_chart(df.set_index("time_seconds")["vehicles_in_frame"])
            st.markdown("#### Congestion category frequency")
            st.bar_chart(pd.Series(congestion_counts, name="Analyzed frames"))

            st.markdown("#### Frame-by-frame report")
            st.dataframe(df, use_container_width=True)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download traffic report (CSV)", csv_bytes,
                               file_name="traffic_analysis_report.csv", mime="text/csv")
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download annotated video", f.read(),
                                   file_name="annotated_traffic_video.mp4", mime="video/mp4")
            st.success("Analysis completed.")
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Check that the video is valid and that the model name/path is correct.")
else:
    st.info("Upload a traffic video to begin. Use a clear, fixed-camera road video for best results.")
    st.markdown("""
    **What this prototype does**
    - Detects cars, motorcycles, buses and trucks using a pretrained YOLO model.
    - Tracks detected vehicles across frames with ByteTrack.
    - Estimates traffic congestion using configurable vehicle-count thresholds.
    - Produces an annotated video and downloadable CSV report.
    """)
