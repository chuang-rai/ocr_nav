from PIL import Image, ImageOps


def load_image(image_path: str) -> Image.Image:

    try:
        image = Image.open(image_path)

        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image

    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None
