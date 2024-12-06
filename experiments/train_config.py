# experiments/train_config.py
import os

class TrainConfig:
    def __init__(self):
        self.project_root = "/home/zehra.korkusuz/ClipBased-SyntheticImageDetection"
        self.train_csv = os.path.join(self.project_root, "data/train.csv")
        self.val_csv = os.path.join(self.project_root, "data/validation.csv")
        self.base_dir = os.path.join(self.project_root, "data")
        self.test_set_dir = os.path.join(self.base_dir, "test_set")  
        self.output_dir = os.path.join(self.project_root, "outputs")
        self.batch_size = 2
        self.device = "cuda"
        self.debug = True

        # Verify paths exist
        self._verify_paths()

    def _verify_paths(self):
        required_files = [self.train_csv, self.val_csv]
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")