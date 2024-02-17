import os
import face_recognition
import PIL
import numpy as np
import hashlib

from utils.describe import describe_image_blip
from utils.crop import crop_body, crop_face


# check if output dir exists, otherwise create it
if not os.path.exists("output"):
    os.makedirs("output")

input_dir = "input"
output_dir = "output"
target = face_recognition.load_image_file(os.path.join(input_dir, "target.jpg"))
target_face = face_recognition.face_encodings(target)
target_ident = hashlib.sha1(target.tobytes()).hexdigest()[:12]


for file_no, file in enumerate(os.listdir(input_dir)):
    print(file)

    # load the image based on face_recognition.load_image_file()
    # Image top left is (0, 0)
    pil_image = PIL.Image.open(os.path.join(input_dir, file))
    rgb_image = np.array(pil_image.convert("RGB"))
    height, width = rgb_image.shape[0], rgb_image.shape[1]

    if height > width:
        pil_image.resize((int(1024 * width / height), 1024)).save(
            f"output/{file_no:02d}_full.jpg"
        )
    else:
        pil_image.resize((1024, int(1024 * height / width))).save(
            f"output/{file_no:02d}_full.jpg"
        )
    description = f"{target_ident} {describe_image_blip(pil_image)}"
    with open(f"output/{file_no:02d}_full.txt", "w") as f:
        f.write(description)

    # find all face locations in the image
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(
        rgb_image, known_face_locations=face_locations, num_jitters=1, model="large"
    )

    similarity = [
        face_recognition.face_distance(f, target_face)[0] for f in face_encodings
    ]
    if similarity:
        ix = similarity.index(min(similarity))
        if similarity[ix] >= 0.6:
            continue

        face_location = face_locations[ix]
        body_crop_pil_image = crop_body(rgb_image, face_location)
        if body_crop_pil_image is not None:
            body_crop_pil_image.save(
                os.path.join(output_dir, f"{file_no:02d}_body.jpg")
            )
            body_crop_description = (
                f"{target_ident} {describe_image_blip(body_crop_pil_image)}"
            )
            with open(os.path.join(output_dir, f"{file_no:02d}_body.txt"), "w") as f:
                f.write(body_crop_description)

        face_crop_pil_image = crop_face(rgb_image, face_location)
        if face_crop_pil_image is not None:
            face_crop_pil_image.save(
                os.path.join(output_dir, f"{file_no:02d}_face.jpg")
            )
            face_crop_description = (
                f"{target_ident} {describe_image_blip(face_crop_pil_image)}"
            )
            with open(os.path.join(output_dir, f"{file_no:02d}_face.txt"), "w") as f:
                f.write(face_crop_description)
