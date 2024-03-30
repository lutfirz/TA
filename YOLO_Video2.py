import torch
import numpy as np
import cv2

import sort
import utilities
import homography_tracker

def video_detection(path_x, path_y):
    video1 = cv2.VideoCapture(path_x)
    video2 = cv2.VideoCapture(path_y)

    # 1000 was choosen arbitrarily
    feat_detector = cv2.SIFT_create(1000)

    _, frame1 = video1.read()
    _, frame2 = video2.read()

    kpts1, des1 = feat_detector.detectAndCompute(frame1, None)
    kpts2, des2 = feat_detector.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher()

    # NOTE: k=2 means the euclidian distance between the two closest matches
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cam4_H_cam1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    np.save("matrix.npy", cam4_H_cam1)

    video1 = cv2.VideoCapture(path_x)
    video2 = cv2.VideoCapture(path_y)

    cam4_H_cam1 = np.load("matrix.npy")
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

    homographies = list()
    homographies.append(np.eye(3))
    homographies.append(cam1_H_cam4)

    detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
    detector.agnostic = True

    # Class 0 is Person
    detector.classes = [0]
    detector.conf = 0.3

    trackers = [
        sort.Sort(
            max_age=30, min_hits=3, iou_threshold=0.3
        )
        for _ in range(2)
    ]
    global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)

    num_frames1 = video1.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames2 = video2.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = min(num_frames2, num_frames1)
    num_frames = int(num_frames)

    # NOTE: Second video 'cam4.mp4' is 17 frames behind the first video 'cam1.mp4'
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    video = None
    for idx in range(num_frames):
        # Get frames
        frame1 = video1.read()[1]
        frame2 = video2.read()[1]

        # NOTE: YoloV5 expects the images to be RGB instead of BGR
        frames = [frame1[:, :, ::-1], frame2[:, :, ::-1]]

        anno = detector(frames)

        dets, tracks = [], []
        for i in range(len(anno)):
            # Sort Tracker requires (x1, y1, x2, y2) bounding box shape
            det = anno.xyxy[i].cpu().numpy()
            det[:, :4] = np.intp(det[:, :4])
            dets.append(det)

            # Updating each tracker measures
            tracker = trackers[i].update(det[:, :4], det[:, -1])
            tracks.append(tracker)

        global_ids = global_tracker.update(tracks)

        for i in range(2):
            frames[i] = utilities.draw_tracks(
                frames[i][:, :, ::-1],
                tracks[i],
                global_ids[i],
                i,
                classes=detector.names,
            )
        vis = np.hstack(frames)
        yield vis
cv2.destroyAllWindows()