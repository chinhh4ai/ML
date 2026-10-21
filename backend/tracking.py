"""Class-aware greedy IoU association. Local IDs, not guaranteed individual counts."""


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


class Tracker:
    def __init__(self, max_age=3, threshold=.15):
        self.tracks = {}
        self.next_id = 1
        self.max_age = max_age
        self.threshold = threshold

    def update(self, objects, timestamp):
        self.tracks = {k: t for k, t in self.tracks.items() if timestamp - t['time'] <= self.max_age}
        candidates = sorted([(iou(obj['box'], track['box']), i, tid)
                             for i, obj in enumerate(objects) for tid, track in self.tracks.items()
                             if obj['label'] == track['label']], reverse=True)
        used_objects, used_tracks = set(), set()
        for score, index, tid in candidates:
            if score < self.threshold or index in used_objects or tid in used_tracks:
                continue
            objects[index]['track_id'] = tid
            used_objects.add(index)
            used_tracks.add(tid)
        for index, obj in enumerate(objects):
            if index not in used_objects:
                obj['track_id'] = self.next_id
                self.next_id += 1
            self.tracks[obj['track_id']] = {'label': obj['label'], 'box': obj['box'], 'time': timestamp}
        return objects
