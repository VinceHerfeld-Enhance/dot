from typing import List, Tuple
import torch
from .dense_optical_tracking import DenseOpticalTracker
from .properties import CHECKPOINT_ROOT, CONFIG_ROOT

class DOTSOpticalFlow(DenseOpticalTracker):
    def __init__(self, width=128, height=128):
        super(DOTSOpticalFlow, self).__init__(
            width=width,
            height=height,
            tracker_config=CONFIG_ROOT / "cotracker2_patch_4_wind_8.json",
            tracker_path= CHECKPOINT_ROOT / "movi_f_cotracker2_patch_4_wind_8.pth",
            estimator_config=CONFIG_ROOT / "raft_patch_8.json",
            estimator_path=CHECKPOINT_ROOT / "cvo_raft_patch_8.pth",
            refiner_config=CONFIG_ROOT / "raft_patch_4_alpha.json",
            refiner_path=CHECKPOINT_ROOT / "movi_f_raft_patch_4_alpha.pth"
        )
        self.height = height
        self.width = width

    def forward(self, ref: torch.Tensor, frames:torch.Tensor) -> List[torch.Tensor]:
        """ Estimate optical flow from each frame to a reference frame, by batch. 
        
        Args:
            ref: (N, C, H, W) Reference frame
            frames: (N, C, H, W) Video frames
        """
        video = torch.cat([ref.unsqueeze(1), frames.unsqueeze(1)], dim=1)
        data = {"video": video}
        flow, alpha = super().forward(data, mode="flow_from_last_to_first_frame")
        flow = flow.permute(0, 3, 1, 2)
        print(f"Max flow: {flow.max()}, Min flow: {flow.min()}")
        return [flow]
    
    def set_image_size(self, width:int, height:int):
        self.__init__(width=width, height=height)