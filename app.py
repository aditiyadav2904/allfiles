import io
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI(tilte="mnist")
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define same model architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn_weights.pkl", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.get("/")
def home():
    return {"message": "MNIST CNN API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess
    image = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {"prediction": int(predicted.item())}
