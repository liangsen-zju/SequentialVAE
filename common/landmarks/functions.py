import torch


# target, torch.Tensor, (Bx2x16, 68, 2)
def get_mouth_width(landmark):
    """ Get mouth width between L48 and L54
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        tensor, (N,)
    """

    WM = (landmark[:, 48, :] - landmark[:, 54, :])  * (landmark[:, 48, :] - landmark[:, 54, :])   # (N, 2)
    WM = torch.sqrt(WM[:, 0] + WM[:, 1])

    return WM

def get_mouth_open_param(landmark, eps=1e-6):
    """ Get mouth open distance and rate
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        two tensors: lip distance  and open rate (N,), (N, )
    """
    up = 0.5 * (landmark[:, 50, :] + landmark[:, 52, :])
    dw = 0.5 * (landmark[:, 56, :] + landmark[:, 58, :])

    distance = (up - dw) * (up -dw)  # (N, 2)
    distance = torch.sqrt(distance[:, 0] + distance[:, 1])  # (N, )
    width = get_mouth_width(landmark)
    rate = distance / (width + eps)
    rate = torch.where(torch.isinf(rate), torch.full_like(rate, 1), rate)  # for bigger rate --> 1

    # assert torch.any(torch.isnan(rate)), f"rate have nan"

    # if torch.any(torch.isnan(rate)) or torch.any(torch.isinf(rate)) :
    #     # print("KKKKKK,",rate)
    #     for i, item in enumerate(rate):
    #         if torch.isnan(item) or torch.isinf(item):
    #             print(f"error mouth nan, item = {item} distance = {distance[i]}, width={width[i]}, data = {landmark[i, [50,52,56,58]]}")
            

    return distance, rate


def get_eye_width(landmark):
    """ Get left eye width between L36 and L39, and right eye width between L42 and L45
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        two tensor of left and right eye width , (N,), (N,)
    """
    return get_left_eye_width(landmark), get_right_eye_width(landmark)

def get_left_eye_width(landmark):
    """ Get left eye width between L36 and L39, and right eye width between L42 and L45
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        two tensor of left and right eye width , (N,), (N,)
    """

    WL = (landmark[:, 36, :] - landmark[:, 39, :])  * (landmark[:, 36, :] - landmark[:, 39, :])   # (N, 2)
    WL = torch.sqrt(WL[:, 0] + WL[:, 1])

    return WL


def get_right_eye_width(landmark):
    """ Get left eye width between L36 and L39, and right eye width between L42 and L45
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        two tensor of left and right eye width , (N,), (N,)
    """
    WR = (landmark[:, 42, :] - landmark[:, 45, :])  * (landmark[:, 42, :] - landmark[:, 45, :])   # (N, 2)
    WR = torch.sqrt(WR[:, 0] + WR[:, 1])

    return WR


def get_eye_open_param(landmark):
    """ Get left/right eye open distance and rate
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        four tensors: left/right eye distance and open rate (N,), (N, )
    """
    L_dist, L_rate = get_left_eye_open_param(landmark)
    R_dist, R_rate = get_right_eye_open_param(landmark)
    
    return L_dist, L_rate, R_dist, R_rate

def get_left_eye_open_param(landmark, eps=1e-6):
    """ Get left eye open distance and rate
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        four tensors: left eye distance and open rate (N,), (N, )
    """
    up = 0.5 * (landmark[:, 37, :] + landmark[:, 38, :])
    dw = 0.5 * (landmark[:, 40, :] + landmark[:, 41, :])

    distance = (up - dw) * (up -dw)  # (N, 2)
    distance = torch.sqrt(distance[:, 0] + distance[:, 1])  # (N, )
    width = get_left_eye_width(landmark)

    rate = distance / (width + eps)
    rate = torch.where(torch.isinf(rate), torch.full_like(rate, 1), rate)  # for bigger rate --> 1

    # if torch.any(torch.isnan(rate)) or torch.any(torch.isinf(rate)) :
    #     # print("KKKKKK,",rate)
    #     for i, item in enumerate(rate):
    #         if torch.isnan(item) or torch.isinf(item):
    #             print(f"error eye nan, item = {item} distance = {distance[i]}, width={width[i]}, data = {landmark[i, 37:42]}")

    return distance, rate


def get_right_eye_open_param(landmark, eps = 1e-6):
    """ Get left eye open distance and rate
    Params:
        landmark: torch, (N, 68, 2)
    Return:
        four tensors: left eye distance and open rate (N,), (N, )
    """
    up = 0.5 * (landmark[:, 43, :] + landmark[:, 44, :])
    dw = 0.5 * (landmark[:, 46, :] + landmark[:, 47, :])

    distance = (up - dw) * (up -dw)  # (N, 2)
    distance = torch.sqrt(distance[:, 0] + distance[:, 1])  # (N, )
    rate = distance / (get_left_eye_width(landmark) + eps)
    rate = torch.where(torch.isinf(rate), torch.full_like(rate, 1), rate)  # for bigger rate --> 1

    return distance, rate