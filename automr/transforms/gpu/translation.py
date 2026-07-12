import torch
import kornia.geometry.transform as K

from .backend import DEVICE


# ==========================================================
# GPU Spatial Translation
# ==========================================================

def shift_right(
    image,
    pixels=5,
    seed=None,
):
    """
    GPU Spatial Translation
    """

    # ----------------------------
    # To Tensor
    # ----------------------------
    if isinstance(image, np.ndarray):

        img = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

    else:

        img = image.unsqueeze(0).to(DEVICE)

    B, C, H, W = img.shape

    pixels = max(1, int(pixels))

    # ----------------------------
    # Random translation
    # ----------------------------
    dx = torch.randint(
        -pixels,
        pixels + 1,
        (1,),
        device=DEVICE,
    ).float()

    dy = torch.randint(
        -pixels,
        pixels + 1,
        (1,),
        device=DEVICE,
    ).float()

    transform = torch.eye(
        3,
        device=DEVICE,
    ).unsqueeze(0)

    transform[:, 0, 2] = dx
    transform[:, 1, 2] = dy

    # ----------------------------
    # GPU Warp
    # ----------------------------
    translated = K.warp_perspective(
        img,
        transform,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )

    # ----------------------------
    # Return
    # ----------------------------
    return (
        translated[0]
        .clamp(0, 255)
        .permute(1, 2, 0)
        .byte()
        .cpu()
        .numpy()
    )