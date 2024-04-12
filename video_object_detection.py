import argparse
import os
import cv2
import torch
import datetime
import time
from imageai.Detection import VideoObjectDetection


def set_environment():
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    torch.cuda.set_device("cuda:0")


def get_detector(execution_path):
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "models/retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()
    return detector


def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")


def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print(
        "Output average count for unique objects in the last second: ",
        average_output_count,
    )
    print("------------END OF A SECOND --------------")


def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print(
        "Output average count for unique objects in the last minute: ",
        average_output_count,
    )
    print("------------END OF A MINUTE --------------")


def detect_objects(detector, camera, run_time):
    start_time = time.time()
    while True:
        if run_time != 0 and time.time() - start_time >= run_time:
            break

        ret, frame = camera.read()
        print(ret)

        if not ret:
            break

        video_path = detector.detectObjectsFromVideo(
            camera_input=camera,
            output_file_path=f"clips/camera_detected_video_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            frames_per_second=24,
            log_progress=True,
            minimum_percentage_probability=0,
            per_second_function=forSeconds,
            per_frame_function=forFrame,
            per_minute_function=forMinute,
        )

        # Draw annotations on frame
        for obj in detector.detected_objects:
            bounding_box = obj["bounding_box"]
            x_min, y_min, x_max, y_max = bounding_box
            label = obj["name"]
            confidence = obj["percentage_probability"]
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f}%)",
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Live Video with Annotations", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


def run_detect_objects(run_time, app):
    set_environment()
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)
    detector = get_detector(execution_path)
    while app.is_running:
        detect_objects(detector, camera, run_time)
    camera.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify the runtime of the script.")
    parser.add_argument(
        "--run_time",
        type=int,
        help="The runtime of the script in seconds. If 0, the script will run until manually interrupted.",
        default=0,
    )
    args = parser.parse_args()
    run_detect_objects(args.run_time)
# Release resources
    cv2.destroyAllWindows()
    
