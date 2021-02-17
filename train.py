import os
from pathlib import Path
from models.ssd import SSD
from src.model.trainer import Trainer
from src.data.utils import collate_fn
from models.metrics import MultiboxLoss
from torch.utils.data import DataLoader
from src.data.dataset import MarketDataset



project_path = Path(__file__).parent
dataset_path = os.path.join(project_path, "data")
train_dataset_path = os.path.join(dataset_path, "train")

train_dataset = MarketDataset(root_dir=train_dataset_path,train=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=32,collate_fn=collate_fn)
val_dataset = MarketDataset(root_dir=train_dataset_path,train=False)
val_loader = DataLoader(dataset=train_dataset,batch_size=32)

ssd_model = SSD("train",300,5)
criterion = MultiboxLoss(num_cls=5,priors=ssd_model.priors)


conf = {
    "number_of_epochs": 50,
    "val_every": 5,
    "optimizer": "SGD",
    "lr": 0.004,
    "device": "cpu",
    "result_path": os.path.join(project_path,"results"),
    "batch_size": 5
}

trainer = Trainer(model=ssd_model, criteria=criterion,train_loader=train_loader,val_loader=val_loader,config=conf)

trainer.train_supervised()
