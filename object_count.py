import cv2
from ultralytics import YOLO

# Load YOLOv8 model
yolo = YOLO('yolov8s.pt')

# Open the webcam (use 0 for default)
vd = cv2.VideoCapture(0)

while True:
    ret, frame = vd.read()
    if not ret or frame is None:
        print("❌ Failed to grab frame")
        break

    # Run YOLO tracking
    results = yolo.track(frame, stream=True)

    # Initialize counters
    object_count = 0
    person_count = 0

    for result in results:
        class_names = result.names

        for box in result.boxes:
            object_count += 1  # Count every object
            class_id = int(box.cls[0])

            # Count persons
            if class_names[class_id] == 'person':
                person_count += 1

            # Draw box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{class_names[class_id]} {box.conf[0]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show counts on frame
    cv2.putText(frame, f'Total objects: {object_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f'Persons: {person_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 100, 255), 2)

    # Display frame
    cv2.imshow('frame', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vd.release()
cv2.destroyAllWindows()