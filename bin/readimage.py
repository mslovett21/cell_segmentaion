import torchvision
from skimage import io
from skimage.color import gray2rgb

def readimage(image_path):
    """
    Takes a raw 512 x 512 png and prepares it for the model
    """
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((256, 256)),
        ]
    )
    cell_img_org = io.imread(image_path)
    cell_img = gray2rgb(cell_img_org)
    cell_img = train_transforms(cell_img)
    return cell_img
