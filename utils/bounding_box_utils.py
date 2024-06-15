def get_center_of_bounding_box(bounding_box):
    x1, y1, x2, y2 = bounding_box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), y2


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    keypoint_index = keypoint_indices[0]
    for keypoint_indice in keypoint_indices:
        keypoint = keypoints[keypoint_indice * 2], keypoints[keypoint_indice * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            keypoint_index = keypoint_indice

    return keypoint_index


def get_bounding_box_height(bounding_box):
    return bounding_box[3] - bounding_box[1]


def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_bbox_center(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
