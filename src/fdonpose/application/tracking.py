"""State of one live pose session; no camera, UI or CUDA imports."""


class TrackingSession:
    def __init__(self, estimator, synchronize):
        self.estimator = estimator
        self.synchronize = synchronize
        self.confirmed_frame = None
        self._confirmed_state = None
        self._register_at = 0.0

    def seek_action(self, now):
        if now >= self._register_at:
            return "register"
        return "recover" if self._confirmed_state is not None else "wait"

    def begin_registration(self, now):
        # Throttle even if segmentation/depth checks reject the candidate.
        self._register_at = now + 5.0

    def register(self, rgb, depth, K, mask, iterations):
        pose = self.estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=iterations)
        self.synchronize()
        return pose

    def track(self, rgb, depth, K, *, recover=False):
        if recover:
            self.estimator.pose_last = self._confirmed_state.clone()
        pose = self.estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=4 if recover else 2)
        self.synchronize()
        return pose

    def confirm(self, rgb, depth, number, timestamp, pose):
        self.confirmed_frame = rgb, depth, number, timestamp, pose.copy()
        self._confirmed_state = self.estimator.pose_last.clone()

    def tracking_lost(self, now):
        self.estimator.pose_last = None
        self._register_at = now + 3.0
