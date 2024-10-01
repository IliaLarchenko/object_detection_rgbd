import cv2


def draw_annotations(cv_image, detections, class_colors):
    """Draws object detection annotations on an image."""

    annotated_image = cv_image.copy()
    for detection in detections.detections:
        bbox = detection.bbox
        class_id = detection.results[0].hypothesis.class_id
        score = detection.results[0].hypothesis.score

        color = class_colors.get(class_id, (255, 255, 255))
        if isinstance(color, list):
            color = tuple(color)

        cx = bbox.center.position.x
        cy = bbox.center.position.y
        w = bbox.size_x
        h = bbox.size_y

        xmin = int(cx - w / 2)
        ymin = int(cy - h / 2)
        xmax = int(cx + w / 2)
        ymax = int(cy + h / 2)

        cv2.rectangle(
            annotated_image, (xmin, ymin), (xmax, ymax), color=color, thickness=2
        )

        cv2.putText(
            annotated_image,
            f"{class_id}: {score:.2f}",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    return annotated_image
