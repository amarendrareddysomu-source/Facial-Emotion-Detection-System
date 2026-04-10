from deep_sort_realtime.deepsort_tracker import DeepSort

class HumanTracker:
    def __init__(self, max_age=30):
        # max_age: Maximum number of missed misses before a track is deleted
        # Adding nms_max_overlap=1.0 suppresses built-in NMS which we don't need after YOLO
        self.tracker = DeepSort(max_age=max_age, n_init=1, nms_max_overlap=1.0)

    def update(self, detections, frame):
        # Update tracker with detections
        tracks = self.tracker.update_tracks(detections, frame=frame)
        active_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb() # Left, Top, Right, Bottom
            active_tracks.append((track_id, ltrb))
        return active_tracks
