import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
from params import audio_model_path, pred_mel_path


def predict_audio():
    model = timm.create_model('convnext_base', pretrained=False, num_classes=2)

    model_load_path = audio_model_path
    model.load_state_dict(torch.load(model_load_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((400, 1000)),
        transforms.ToTensor()
    ])

    image = Image.open(pred_mel_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    return outputs.cpu().numpy()
