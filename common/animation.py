import moviepy.editor as mpy

import sys
sys.path.insert(0, "/mnt/data/CODE/Audio2Expression")
from common.landmark import landmark_to_images


def save_animation(data_motion, path_save, suffix, w=128*5, h=72*5, dpi=80):
    """ convert data_motion to .mp4 animation
    data_motion, shape [B, N, nc_motion1, nc_motion2, ...]
    """
    # no need invert
    # anim_clips = inv_standardize(data_motion[:self.nseq_test, :, :],  self.motion_scaler)
    fps = 30
    B, *_ = data_motion.shape

    for i in range(B):
        idata_motion = data_motion[i, :, :].reshape(-1, 68, 2)
        images_pt = landmark_to_images(idata_motion, w=w, h=h, dpi=dpi, norm=True, prex="PT")  

        duration = idata_motion.shape[0] / fps
        # animation = mpy.VideoClip(lambda t: images_pt[int(fps * t + 0.5) - 1,...], duration= duration )
        animation = mpy.VideoClip(lambda t: images_pt[int(fps * t + 0.5),...], duration= duration )
        
        ipath_save = path_save.joinpath(f"{suffix}_{i:02d}.mp4")
        animation.write_videofile(str(ipath_save), fps=fps, logger=None)
        