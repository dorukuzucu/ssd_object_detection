import os
import pandas as pd
from pathlib import Path
import shutil


def split_dataset(path,train_test_split=0.85):
    project_dir = os.path.join(Path(__file__).parents[2],"data")
    print(project_dir)


split_dataset("ss")

