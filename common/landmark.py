import os
import cv2
import torch
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

plt.rcParams['axes.facecolor'] = 'black'

def landmark_normalize(ldmk):
    # ldmk shape = (68, 2)
    
    pos_mean = np.mean(ldmk, axis=0)  # (2, )
    pos_max = np.max(ldmk, axis=0)    # (2, )
    pos_min = np.min(ldmk, axis=0)    # (2, )
    
    side = int(1.5 * np.max(pos_max - pos_min))   # (1, )
    # print(f"landmarks = {ldmk[::10, :]}, \n mean={pos_mean}, max={pos_max}, min={pos_min}, side={side}")

    tran = pos_mean - 0.5 * side             # (2,   )
    ldmk_tran = ldmk - tran[np.newaxis, :]   # (68, 2)
    ldmk_norm = ldmk_tran / side             # (68, 2)

    # print(f"ldmk_tran = {ldmk_tran[::10, :]}, \n ldmk_tran = {ldmk_norm[::10, :]}")

    return ldmk_norm, pos_mean, side


def landmark_normalize_by_param(ldmk, params):
    # ldmk shape = (B, 68, 2)
    # params shape = (B, 3)

    pos_mean = params[:, 0:2 ]              # (B, 2)
    pos_mean = pos_mean[:, np.newaxis, :]   # (B, 1, 2)
    side = params[:, 2:]                    # (B, 2)
    side = side[:, np.newaxis, :]           # (B, 1, 2)

    tran = pos_mean - 0.5 * side            # (B, 1, 2)
    ldmk_norm = (ldmk - tran) / side        # (B, 68, 2)

    return ldmk_norm

def landmark_to_images(ldmks, w=256, h=256, dpi=100, norm=True, prex=None):
    # landmarks shape = (nbatch, 68, 2)
    nframe = ldmks.shape[0]

    v_images = []
    for i in range(nframe):
        ildmk = ldmks[i, :, :]
        text = f"{prex}-frame-{i:04d}"
        v_images.append(landmark_to_image(ildmk, w, h, dpi, norm, text))

    return np.asarray(v_images)


def landmark_to_image(ildmk, w=256, h=256, dpi=100, norm=True, text=None):
    # landmarks shape = (68, 2)

    if norm:
        ildmk[:, 0] = w - ildmk[:, 0] * w
        ildmk[:, 1] = h - ildmk[:, 1] * h
        
    else:
        # ildmk[:, 0] = w - ildmk[:, 0]
        ildmk[:, 1] = h - ildmk[:, 1]

    if isinstance(ildmk, torch.Tensor):
        ildmk = ildmk.cpu().numpy()

    # plot 
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    fig.patch.set_facecolor('w')

    ax = fig.add_subplot(1,1,1)
    # ax.imshow(np.zeros(shape=(w, h)))

    ax.set_facecolor('m')
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # ax.scatter(ildmk[:, 0], ildmk[:, 1],  marker='o', s=5, color="#808080",alpha=1)
    ax.scatter(ildmk[ 0:17,0], ildmk[ 0:17,1],  marker='o', s=3, color="#008B8B",alpha=1)
    ax.scatter(ildmk[17:22,0], ildmk[17:22,1],  marker='o', s=3, color="#F4A460",alpha=1)
    ax.scatter(ildmk[22:27,0], ildmk[22:27,1],  marker='o', s=3, color="#F4A460",alpha=1)
    ax.scatter(ildmk[27:36,0], ildmk[27:36,1],  marker='o', s=3, color="blue",alpha=1)
    ax.scatter(ildmk[36:48,0], ildmk[36:48,1],  marker='o', s=3, color="#DC143C",alpha=1)
    ax.scatter(ildmk[48:60,0], ildmk[48:60,1],  marker='o', s=3, color="#DC143C",alpha=1)
    ax.scatter(ildmk[60:68,0], ildmk[60:68,1],  marker='o', s=3, color="#FF1493",alpha=1)

    #chin
    ax.plot(ildmk[ 0:17,0], ildmk[ 0:17,1], marker='', markersize=1, linestyle='-', color='#008B8B', lw=2, alpha=1)
    
    #left and right eyebrow
    ax.plot(ildmk[17:22,0], ildmk[17:22,1], marker='', markersize=1, linestyle='-', color='#F4A460', lw=2, alpha=1)
    ax.plot(ildmk[22:27,0], ildmk[22:27,1], marker='', markersize=1, linestyle='-', color='#F4A460', lw=2, alpha=1)
    
    #nose
    ax.plot(ildmk[27:31,0], ildmk[27:31,1], marker='', markersize=1, linestyle='-', color='blue', lw=2, alpha=1)
    ax.plot(ildmk[31:36,0], ildmk[31:36,1], marker='', markersize=1, linestyle='-', color='blue', lw=2, alpha=1)

    #left and right eye
    ax.plot(ildmk[36:42,0], ildmk[36:42,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[42:48,0], ildmk[42:48,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)

    ax.plot(ildmk[[41,36],0], ildmk[[41,36],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[[47,42],0], ildmk[[47,42],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    
    #outer and inner lip
    ax.plot(ildmk[48:60,0], ildmk[48:60,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[60:68,0], ildmk[60:68,1], marker='', markersize=1, linestyle='-', color='#FF1493', lw=2, alpha=1) 

    ax.plot(ildmk[[59,48],0], ildmk[[59,48],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    ax.plot(ildmk[[67,60],0], ildmk[[67,60],1], marker='', markersize=1, linestyle='-', color='#FF1493', lw=2, alpha=1) 

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')

    if text is not None:
        ax.text(10, 10, text)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buffer = buffer.reshape((h, w, 3))
    image = np.asarray(buffer)
    # print(f"w,h={w},{h}, buffer shape = {buffer.shape}, dtype = {buffer.dtype}, class = {(buffer)}")

    plt.close(fig)

    return image

# def landmark_to_image(ildmk, w=256, h=256, dpi=100, norm=True, text=None):
#     # landmarks shape = (68, 2)

#     if norm:
#         ildmk = w - ildmk * w
#     else:
#         ildmk[:, 1] = h - ildmk[:, 1]

#     if isinstance(ildmk, torch.Tensor):
#         ildmk = ildmk.cpu().numpy()

#     # plot 
#     fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
#     fig.patch.set_facecolor('w')

#     ax = fig.add_subplot(1,1,1)
#     # ax.imshow(np.zeros(shape=(w, h)))

#     ax.set_facecolor('m')
#     # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     # ax.scatter(ildmk[:, 0], ildmk[:, 1],  marker='o', s=5, color="#808080",alpha=1)
#     ax.scatter(ildmk[ 0:17,0], ildmk[ 0:17,1],  marker='o', s=3, color="#008B8B",alpha=1)
#     ax.scatter(ildmk[17:22,0], ildmk[17:22,1],  marker='o', s=3, color="#F4A460",alpha=1)
#     ax.scatter(ildmk[22:27,0], ildmk[22:27,1],  marker='o', s=3, color="#F4A460",alpha=1)
#     ax.scatter(ildmk[27:36,0], ildmk[27:36,1],  marker='o', s=3, color="blue",alpha=1)
#     ax.scatter(ildmk[36:48,0], ildmk[36:48,1],  marker='o', s=3, color="#DC143C",alpha=1)
#     ax.scatter(ildmk[48:60,0], ildmk[48:60,1],  marker='o', s=3, color="#DC143C",alpha=1)
#     ax.scatter(ildmk[60:68,0], ildmk[60:68,1],  marker='o', s=3, color="#FF1493",alpha=1)

#     #chin
#     ax.plot(ildmk[ 0:17,0], ildmk[ 0:17,1], marker='', markersize=1, linestyle='-', color='#008B8B', lw=2, alpha=1)
    
#     #left and right eyebrow
#     ax.plot(ildmk[17:22,0], ildmk[17:22,1], marker='', markersize=1, linestyle='-', color='#F4A460', lw=2, alpha=1)
#     ax.plot(ildmk[22:27,0], ildmk[22:27,1], marker='', markersize=1, linestyle='-', color='#F4A460', lw=2, alpha=1)
    
#     #nose
#     ax.plot(ildmk[27:31,0], ildmk[27:31,1], marker='', markersize=1, linestyle='-', color='blue', lw=2, alpha=1)
#     ax.plot(ildmk[31:36,0], ildmk[31:36,1], marker='', markersize=1, linestyle='-', color='blue', lw=2, alpha=1)

#     #left and right eye
#     ax.plot(ildmk[36:42,0], ildmk[36:42,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
#     ax.plot(ildmk[42:48,0], ildmk[42:48,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)

#     ax.plot(ildmk[[41,36],0], ildmk[[41,36],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
#     ax.plot(ildmk[[47,42],0], ildmk[[47,42],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
    
#     #outer and inner lip
#     ax.plot(ildmk[48:60,0], ildmk[48:60,1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
#     ax.plot(ildmk[60:68,0], ildmk[60:68,1], marker='', markersize=1, linestyle='-', color='#FF1493', lw=2, alpha=1) 

#     ax.plot(ildmk[[59,48],0], ildmk[[59,48],1], marker='', markersize=1, linestyle='-', color='#DC143C', lw=2, alpha=1)
#     ax.plot(ildmk[[67,60],0], ildmk[[67,60],1], marker='', markersize=1, linestyle='-', color='#FF1493', lw=2, alpha=1) 

#     ax.set_xlim(0, w)
#     ax.set_ylim(0, h)
#     ax.axis('off')

#     if text is not None:
#         ax.text(10, 10, text)

#     fig.canvas.draw()
#     w, h = fig.canvas.get_width_height()

#     buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     buffer = buffer.reshape((h, w, 3))
#     image = np.asarray(buffer)
#     # print(f"w,h={w},{h}, buffer shape = {buffer.shape}, dtype = {buffer.dtype}, class = {(buffer)}")

#     plt.close(fig)

#     return image

def landmark_to_mask(ildmk, w=256, h=256, dpi=100, norm=True):
    # landmarks shape = (68, 2)
    # print(f"landmark data shape = {ildmk.shape}")
    if norm:
        ildmk = ildmk * w

    # plot 
    fig = plt.figure(figsize=(h/dpi, w/dpi), dpi=dpi)
    ax = fig.add_subplot(1,1,1)

    # ax.imshow(np.ones(shape=(w, h)))
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    #chin
    ax.plot(ildmk[ 0:17,0], ildmk[ 0:17,1], marker='', markersize=1, linestyle='-', color='green', lw=2)
    #left and right eyebrow
    ax.plot(ildmk[17:22,0], ildmk[17:22,1], marker='', markersize=1, linestyle='-', color='orange', lw=2)
    ax.plot(ildmk[22:27,0], ildmk[22:27,1], marker='', markersize=1, linestyle='-', color='orange', lw=2)
    #nose
    ax.plot(ildmk[27:31,0], ildmk[27:31,1], marker='', markersize=1, linestyle='-', color='blue', lw=2)
    ax.plot(ildmk[31:36,0], ildmk[31:36,1], marker='', markersize=1, linestyle='-', color='blue', lw=2)
    #left and right eye
    ax.plot(ildmk[36:42,0], ildmk[36:42,1], marker='', markersize=1, linestyle='-', color='red', lw=2)
    ax.plot(ildmk[42:48,0], ildmk[42:48,1], marker='', markersize=1, linestyle='-', color='red', lw=2)
    #outer and inner lip
    ax.plot(ildmk[48:60,0], ildmk[48:60,1], marker='', markersize=1, linestyle='-', color='purple', lw=2)
    ax.plot(ildmk[60:68,0], ildmk[60:68,1], marker='', markersize=1, linestyle='-', color='pink', lw=2) 
    ax.axis('off')


    fig.canvas.draw()


    buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # buffer.shape = (w, h, 4)

    image = buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image







