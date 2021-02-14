from torch.utils.data import DataLoader

from models.ssd import SSD
from models.metrics import MultiboxLoss
from src.data.dataset import MarketDataset
from pathlib import Path
import os

project_path = Path(__file__).parent

model = SSD("train",300,5)
train_dataset_path = os.path.join(project_path, "data", "train")
train_dataset = MarketDataset(root_dir=train_dataset_path)
train_loader = DataLoader(dataset=train_dataset,batch_size=1)
criterion = MultiboxLoss(num_cls=5,priors=model.priors)

for img,gt in train_loader:


    out = model(img)

    loss = criterion.calculate(out,gt)
    break

print("FWD Done")
