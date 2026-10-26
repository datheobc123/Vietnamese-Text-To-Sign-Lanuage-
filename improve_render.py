from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
import numpy as np

class PoseVisualizerCustom(PoseVisualizer):
    def __init__(self, pose, out_w=640, out_h=360, thickness=None):
        super().__init__(pose, thickness)
        self.out_w = out_w
        self.out_h = out_h
        
        self.orig_w = self.pose.header.dimensions.width
        self.orig_h = self.pose.header.dimensions.height
        
        self.scale_x = self.out_w / self.orig_w if self.orig_w > 0 else 1.0
        self.scale_y = self.out_h / self.orig_h if self.orig_h > 0 else 1.0
        
        data = self.pose.body.data.data
        
        num_dims = data.shape[-1]
        scale_factors = np.ones(num_dims)
        scale_factors[0] = self.scale_x
        scale_factors[1] = self.scale_y
        
        scaled_data = data * scale_factors
        self.scaled_data_int = np.array(np.around(scaled_data), dtype="int32")
    
    def draw(self, background_color=(255, 255, 255), max_frames=None):
        if max_frames is None:
            max_frames = len(self.scaled_data_int)
        
        background = np.full(
            (self.out_h, self.out_w, 3),
            fill_value=background_color,
            dtype="uint8"
        )
        
        for frame_idx, (frame, confidence) in enumerate(zip(self.scaled_data_int, self.pose.body.confidence)):
            if frame_idx >= max_frames:
                break
            yield self._draw_frame(frame, confidence, img=background.copy())
