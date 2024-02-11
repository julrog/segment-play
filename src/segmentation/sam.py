from typing import Any, Optional

import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from segmentation.base import Segmentation


class Sam(Segmentation):
    def __init__(
            self, checkpoint: str = 'models/sam_vit_b_01ec64.pth') -> None:
        model_type = 'vit_b'
        device = 'cuda'
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)  # type: ignore
        self.predictor: SamPredictor = SamPredictor(sam)

    def get_image_embedding(self) -> Any:
        return self.predictor.get_image_embedding().cpu().numpy()

    def bbox_masks(
        self,
        bb: np.ndarray,
        points: Optional[np.ndarray] = None,
        point_modes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        point_coords = None
        point_labels = None
        if points is not None and points.any():
            labels = np.ones(points.shape[0])
            if point_modes is not None:
                labels = point_modes
            point_labels = labels
            point_coords = points

        masks, _, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=bb[None, :],
            multimask_output=False,
        )
        masks = masks > self.predictor.model.mask_threshold
        return masks
