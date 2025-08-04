# task2_leaf_count_prediction.py
# Complete code for Task 2: Leaf Count Prediction

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# VISION TRANSFORMER MODEL COMPONENTS
# ============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        
        return x, attn

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class GromoVisionTransformer(nn.Module):
    def __init__(self, input_channels, patch_size, num_patches, projection_dim, 
                 num_heads, num_layers, mlp_dim, num_images, dropout_rate=0.1):
        super().__init__()
        
        self.num_images = num_images
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.patch_embeds = nn.ModuleList([
            PatchEmbedding(224, patch_size, 3, projection_dim)
            for _ in range(num_images)
        ])

        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, projection_dim) * 0.02)
        self.image_encoding = nn.Parameter(torch.randn(1, num_images, projection_dim) * 0.02)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, 4.0, dropout_rate)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(projection_dim)
        self.cross_attention = MultiHeadAttention(projection_dim, num_heads, dropout_rate)

        self.mlp_head = nn.Sequential(
            nn.Linear(projection_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim // 2, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]

        patch_embeddings = []
        for i in range(self.num_images):
            img_x = x[:, i*3:(i+1)*3, :, :]
            patch_embed = self.patch_embeds[i](img_x)
            patch_embeddings.append(patch_embed)

        for i in range(self.num_images):
            patch_embeddings[i] = patch_embeddings[i] + self.positional_encoding

        attention_weights = []
        for layer in self.transformer_blocks:
            layer_attention_weights = []
            for i in range(self.num_images):
                patch_embeddings[i], attn_weights = layer(patch_embeddings[i])
                layer_attention_weights.append(attn_weights)
            attention_weights.append(layer_attention_weights)

        for i in range(self.num_images):
            patch_embeddings[i] = self.norm(patch_embeddings[i])

        image_features = []
        for i in range(self.num_images):
            pooled = patch_embeddings[i].mean(dim=1)
            image_features.append(pooled)
        
        image_features = torch.stack(image_features, dim=1)
        image_features = image_features + self.image_encoding

        cross_features, cross_attn = self.cross_attention(image_features)
        global_features = cross_features.mean(dim=1)
        output = self.mlp_head(global_features)

        return output, attention_weights

# ============================================================================
# DATASET CLASS
# ============================================================================

class CropDataset(Dataset):
    def __init__(self, root_dir, csv_file, images_per_level, crop, plants, days,
                 levels=['L1', 'L2', 'L3', 'L4', 'L5'], transform=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            csv_file (str): Path to the CSV file containing ground truth (filename, leaf_count, age).
            images_per_level (int): Number of images to select per level (should be factors of 24).
            crop (str): Crop type (e.g., "radish").
            plants (int): Number of plants (e.g., 4).
            days (int): Number of days (e.g., 59).
            levels (list): List of levels (e.g., ['L1', 'L2', 'L3', 'L4', 'L5']) (fixed).
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.images_per_level = images_per_level
        self.crop = crop
        self.plants_num = plants
        self.max_days = days
        self.levels = levels
        self.transform = transform
        self.image_data = self._load_metadata()
        self.image_paths = self._load_image_paths()

    def _load_metadata(self):
        """Load CSV file into a pandas DataFrame and map filenames to leaf counts and ages."""
        df = pd.read_csv(self.csv_file)
        df["filename"] = df["filename"].astype(str)  # Ensure filenames are strings
        return df.set_index("filename")  # Use filename as the index for quick lookup

    def _select_angles(self):
        """
        Select angles dynamically for a given level.
        """
        images_needed = self.images_per_level
        selected_angles = [i for i in range(0, 360, int(360 / images_needed))]

        initial_angles = [i for i in range(15, selected_angles[1], 15)]
        multiple_selections = [selected_angles]

        for initial_angle in initial_angles:
            selection = [initial_angle]
            while len(selection) < images_needed:
                next_angle = (selection[-1] + int(360 / images_needed)) % 360
                if next_angle not in selection:
                    selection.append(next_angle)
            multiple_selections.append(selection)
        print(multiple_selections)
        return multiple_selections

    def _load_image_paths(self):
        """
        Load image paths for all levels and plants based on the selection of angles.
        """
        image_paths = []
        multiple_selections = self._select_angles()

        for plant in range(1, self.plants_num + 1):
            plant_path = os.path.join(self.root_dir, self.crop, f"p{plant}")
            if not os.path.isdir(plant_path):
                print(f"Plant directory not found: {plant_path}")
                continue
            for day in range(1, self.max_days + 1):
                day_path = os.path.join(self.root_dir, self.crop, f"p{plant}", f"d{day}")
                if not os.path.isdir(day_path):
                    continue
                for selected_angles in multiple_selections:
                    for level in self.levels:
                        level_path = os.path.join(self.root_dir,self.crop, f"p{plant}", f"d{day}", level)
                        level_image_paths = [
                            os.path.join(level_path, f"{self.crop}_p{plant}_d{day}_{level}_{angle}.png")
                            for angle in selected_angles
                        ]
                        filename = os.path.join(self.crop,f"p{plant}", f"d{day}", level,f"{self.crop}_p{plant}_d{day}_{level}_{selected_angles[0]}.png")
                        print(filename)
                        leaf_count = self.image_data.loc[filename, "leaf_count"]
                        # print(level_image_paths)
                        image_paths.append((level_image_paths, leaf_count,day))  # Append day number along with image paths

        print(f"Total samples loaded: {len(image_paths)}")
        # print(f"individual sample size: {len(image_paths[0][0])}")
        return image_paths


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        """
        Get a batch of images from the dataset corresponding to the angles selected.
        """
        images = []
        leaf_count = self.image_paths[idx][1]
        age = self.image_paths[idx][2]
        # print(leaf_count,age)
        all_images= self.image_paths[idx][0]
        # print("length of all images:", len(all_images))
        for img_path in all_images:  # Get the image paths for this sample
            if os.path.isfile(img_path):
                  level_image = Image.open(img_path)
                  if self.transform:
                      level_image = self.transform(level_image)
                  images.append(level_image)
            else:
                    print(f"Path is not a valid file: {img_path}")

        images = torch.cat(images, dim=0)

        return images, torch.tensor(leaf_count, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)  # Return both images and the corresponding day as ground truth


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def train_leaf_count_model(model, train_loader, val_loader, num_epochs, device, learning_rate=8e-5):
    criterion = nn.SmoothL1Loss()
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.03,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,
        T_mult=2,
        eta_min=1e-7
    )
    
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_mae = float('inf')
    
    print(f"Starting leaf count training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_mae_sum = 0.0
        train_count = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch_idx, (images, leaf_counts, ages) in enumerate(progress_bar):
            images = images.to(device)
            targets = leaf_counts.to(device).float()
            
            if random.random() < 0.1:
                targets = targets + torch.randn_like(targets) * 0.05
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs.squeeze(), targets)
            
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 0.0005 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            with torch.no_grad():
                batch_mae = torch.mean(torch.abs(outputs.squeeze() - targets)).item()
                train_mae_sum += batch_mae
            
            train_loss += loss.item()
            train_count += 1
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{batch_mae:.4f}'
            })
        
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for images, leaf_counts, ages in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                targets = leaf_counts.to(device).float()
                
                outputs, _ = model(images)
                loss = criterion(outputs.squeeze(), targets)
                
                batch_mae = torch.mean(torch.abs(outputs.squeeze() - targets)).item()
                val_mae_sum += batch_mae
                
                val_loss += loss.item()
                val_count += 1
        
        avg_train_loss = train_loss / train_count
        avg_val_loss = val_loss / val_count
        avg_train_mae = train_mae_sum / train_count
        avg_val_mae = val_mae_sum / val_count
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)
        
        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            print(f"  New best validation MAE: {best_val_mae:.4f}")
        
        if early_stopping(avg_val_loss, model):
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed! Best validation MAE: {best_val_mae:.4f}")
    return model, train_losses, val_losses, train_maes, val_maes

def evaluate_leaf_count_model(model, test_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    print("Evaluating leaf count model...")
    
    with torch.no_grad():
        for images, leaf_counts, ages in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            targets = leaf_counts.to(device).float()
            
            outputs, _ = model(images)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2, y_true, y_pred

# ============================================================================
# CONFIGURATION
# ============================================================================

CROPS_CONFIG = {
     'wheat': {
        'plants': 5,
        'days': 118,
        'root_path': '/Users/mk/Downloads/training_data/',
        'csv_file': '/Users/mk/Downloads/training_data/wheat_train.csv'  # Your actual train CSV
    },
}

MODEL_CONFIG = {
    'num_images': 4,
    'input_channels': 12,
    'patch_size': 16,
    'num_patches': 196,
    'projection_dim': 128,
    'num_heads': 4,
    'num_layers': 3,
    'mlp_dim': 256,
    'dropout_rate': 0.15
}

TRAINING_CONFIG = {
    'batch_size': 16,
    'num_epochs': 10,
    'learning_rate': 8e-5,
    'validation_split': 0.25,
    'images_per_level': 4
}

# Enhanced transforms for leaf count
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=25),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
        shear=8
    ),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_task2_leaf_count():
    print("TASK 2: LEAF COUNT PREDICTION TRAINING")
    print("="*60)
    
    crop_name = 'wheat'
    config = CROPS_CONFIG[crop_name]
    
    print("Loading dataset...")
    full_dataset = CropDataset(
        root_dir=config['root_path'],
        csv_file=config['csv_file'],
        images_per_level=TRAINING_CONFIG['images_per_level'],
        crop=crop_name,
        plants=config['plants'],
        days=config['days'],
        transform=train_transform
    )
    
    if len(full_dataset) == 0:
        print(f"No data found for {crop_name}. Exiting.")
        return
    
    print(f"Total dataset size: {len(full_dataset)} samples")
    
    val_ratio = TRAINING_CONFIG['validation_split']
    train_size = int((1 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("Creating model...")
    model = GromoVisionTransformer(**MODEL_CONFIG)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    print("Starting training...")
    model, train_losses, val_losses, train_maes, val_maes = train_leaf_count_model(
        model, train_loader, val_loader,
        TRAINING_CONFIG['num_epochs'],
        device,
        TRAINING_CONFIG['learning_rate']
    )
    
    print("Evaluating model...")
    rmse, mae, r2, y_true, y_pred = evaluate_leaf_count_model(model, val_loader, device)
    
    print(f"\nFINAL RESULTS - TASK 2: LEAF COUNT PREDICTION")
    print("="*60)
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Best Training MAE: {min(train_maes):.4f}")
    print(f"Best Validation MAE: {min(val_maes):.4f}")
    
    print("Saving model...")
    os.makedirs('task2', exist_ok=True)
    model_path = f"task2/{crop_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, train_losses, val_losses, train_maes, val_maes

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("GROMO CHALLENGE 2025 - TASK 2: LEAF COUNT PREDICTION")
    print("="*70)
    
    response = input("Do you want to start training Task 2? (y/n): ").lower()
    if response != 'y':
        print("Training cancelled.")
        exit()
    
    try:
        model, train_losses, val_losses, train_maes, val_maes = train_task2_leaf_count()
        
        print("\nTASK 2 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Model saved in task2/ directory")
        print("Ready for inference on test data")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nGenerated files:")
    print("  - task2/mustard_model.pth (trained model)")
    print("\nTask 2 (Leaf Count Prediction) pipeline complete!")