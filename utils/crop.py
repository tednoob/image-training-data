import PIL
from typing import Optional

from math import ceil, floor


def crop_body(
    rgb_image, face_location, max_image_ratio=0.7
) -> Optional[PIL.Image.Image]:
    height, width = rgb_image.shape[0], rgb_image.shape[1]
    top, right, bottom, left = face_location
    fw = right - left
    fh = bottom - top

    top = max(0, int(top - 1 * fh))
    bottom = min(height, int(top + 8 * fh))
    left = max(0, int(left - 1.5 * fw))
    right = min(width, int(right + 1.5 * fw))

    # No point cutting a too large of a picture from the whole image
    if (bottom - top) * (right - left) > max_image_ratio * height * width:
        return None

    crop_rgb_image = rgb_image[top:bottom, left:right]
    crop_pil_image = PIL.Image.fromarray(crop_rgb_image)
    if bottom - top > right - left:
        return crop_pil_image.resize(
            (int(1024 * (right - left) / (bottom - top)), 1024)
        )
    else:
        return crop_pil_image.resize(
            (1024, int(1024 * (bottom - top) / (right - left)))
        )


def crop_face(
    rgb_image, face_location, max_image_ratio=0.7
) -> Optional[PIL.Image.Image]:
    height, width = rgb_image.shape[0], rgb_image.shape[1]
    top, right, bottom, left = face_location
    fw = right - left
    fh = bottom - top
    cx = (left + right) / 2
    cy = (top + bottom) / 2

    top = max(0, int(top - 1 * fh))
    bottom = min(height - 1, int(bottom + 1 * fh))
    left = max(0, int(left - 1 * fw))
    right = min(width - 1, int(right + 1 * fw))
    if bottom - top > right - left:
        cy = round((bottom + top) / 2)
        top = max(0, ceil(cy - (right - left) / 2))
        bottom = min(height - 1, floor(cy + (right - left) / 2))
    else:
        cx = round((right + left) / 2)
        left = max(0, ceil(cx - (bottom - top) / 2))
        right = min(width - 1, floor(cx + (bottom - top) / 2))


    # No point cutting a too large of a picture from the whole image
    if (bottom - top) * (right - left) > max_image_ratio * height * width:
        return None

    crop_rgb_image = rgb_image[top:bottom, left:right]
    crop_pil_image = PIL.Image.fromarray(crop_rgb_image)
    return crop_pil_image.resize((1024, 1024))
