
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
import math
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

#

class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings."""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
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
    """Multi-head self-attention mechanism."""
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
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x = self.proj(x)
        
        return x, attn

class MLP(nn.Module):
    """MLP block for transformer."""
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
    """Transformer block with self-attention and MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights

class GromoVisionTransformer(nn.Module):
    """Enhanced Vision Transformer for GROMO Challenge."""
    def __init__(self, input_channels, patch_size, num_patches, projection_dim, 
                 num_heads, num_layers, mlp_dim, num_images, dropout_rate=0.1):
        super().__init__()
        
        self.num_images = num_images
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim

        # Separate patch embedding layers for each image
        self.patch_embeds = nn.ModuleList([
            PatchEmbedding(224, patch_size, 3, projection_dim)
            for _ in range(num_images)
        ])

        # Positional Encoding (Learnable)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, projection_dim) * 0.02)
        
        # Image-level positional encoding
        self.image_encoding = nn.Parameter(torch.randn(1, num_images, projection_dim) * 0.02)

        # Transformer Encoder Layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, 4.0, dropout_rate)
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(projection_dim)
        
        # Cross-image attention
        self.cross_attention = MultiHeadAttention(projection_dim, num_heads, dropout_rate)

        # Enhanced MLP Head for regression
        self.mlp_head = nn.Sequential(
            nn.Linear(projection_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim // 2, 1)
        )
        
        # Initialize weights
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

        # Step 1: Patch Embedding (Separate for each image)
        patch_embeddings = []
        for i in range(self.num_images):
            # Split the input channels into separate images (3 channels each for RGB)
            img_x = x[:, i*3:(i+1)*3, :, :]  # Shape: (batch_size, 3, height, width)
            patch_embed = self.patch_embeds[i](img_x)  # Apply separate embedding
            patch_embeddings.append(patch_embed)

        # Step 2: Add Positional Encoding
        for i in range(self.num_images):
            patch_embeddings[i] = patch_embeddings[i] + self.positional_encoding

        # Step 3: Transformer Encoder Layers
        attention_weights = []
        for layer in self.transformer_blocks:
            layer_attention_weights = []
            for i in range(self.num_images):
                patch_embeddings[i], attn_weights = layer(patch_embeddings[i])
                layer_attention_weights.append(attn_weights)
            attention_weights.append(layer_attention_weights)

        # Step 4: Apply layer normalization
        for i in range(self.num_images):
            patch_embeddings[i] = self.norm(patch_embeddings[i])

        # Step 5: Pool patches within each image (mean pooling)
        image_features = []
        for i in range(self.num_images):
            pooled = patch_embeddings[i].mean(dim=1)  # Shape: (batch_size, projection_dim)
            image_features.append(pooled)
        
        # Stack image features: (batch_size, num_images, projection_dim)
        image_features = torch.stack(image_features, dim=1)
        
        # Add image-level positional encoding
        image_features = image_features + self.image_encoding

        # Step 6: Cross-image attention
        cross_features, cross_attn = self.cross_attention(image_features)
        
        # Step 7: Global pooling across images
        global_features = cross_features.mean(dim=1)  # Shape: (batch_size, projection_dim)

        # Step 8: MLP Head for regression
        output = self.mlp_head(global_features)

        return output, attention_weights


class CropDataset(Dataset):
    def __init__(self, root_dir, csv_file, images_per_level, crop, plants, days,
                 levels=['L1', 'L2', 'L3', 'L4', 'L5'], transform=None):
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
        try:
            df = pd.read_csv(self.csv_file)
            df["filename"] = df["filename"].astype(str)
            df.columns = df.columns.str.lower()
            return df.set_index("filename")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def _select_angles(self):
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
        return multiple_selections

    def _load_image_paths(self):
        image_paths = []
        multiple_selections = self._select_angles()

        crop_path = os.path.join(self.root_dir, self.crop)
        if not os.path.exists(crop_path):
            print(f"Crop directory not found: {crop_path}")
            return image_paths
            
        actual_plants = [d for d in os.listdir(crop_path) if d.startswith('p') and os.path.isdir(os.path.join(crop_path, d))]
        actual_plants.sort()

        for plant_dir in actual_plants[:self.plants_num]:
            plant_path = os.path.join(crop_path, plant_dir)
            actual_days = [d for d in os.listdir(plant_path) if d.startswith('d') and os.path.isdir(os.path.join(plant_path, d))]
            actual_days.sort(key=lambda x: int(x[1:]))
            
            for day_dir in actual_days[:self.max_days]:
                day_num = int(day_dir[1:])
                for selected_angles in multiple_selections:
                    for level in self.levels:
                        level_path = os.path.join(plant_path, day_dir, level)
                        if not os.path.exists(level_path):
                            continue
                            
                        level_image_paths = [
                            os.path.join(level_path, f"{self.crop}_{plant_dir}_{day_dir}_{level}_{angle}.png")
                            for angle in selected_angles
                        ]
                        
                        filename = f"{self.crop}/{plant_dir}/{day_dir}/{level}/{self.crop}_{plant_dir}_{day_dir}_{level}_{selected_angles[0]}.png"
                        
                        if filename in self.image_data.index:
                            try:
                                leaf_count = self.image_data.loc[filename, "leaf_count"]
                                image_paths.append((level_image_paths, leaf_count, day_num))
                            except KeyError:
                                continue

        print(f"Total samples loaded: {len(image_paths)}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        images = []
        leaf_count = self.image_paths[idx][1]
        age = self.image_paths[idx][2]
        all_images = self.image_paths[idx][0]
        
        for img_path in all_images:
            if os.path.isfile(img_path):
                try:
                    level_image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        level_image = self.transform(level_image)
                    images.append(level_image)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

        target_images = 4
        if len(images) < target_images:
            while len(images) < target_images:
                if len(images) > 0:
                    images.append(images[-1].clone())
                else:
                    images.append(torch.zeros(3, 224, 224))
        elif len(images) > target_images:
            images = images[:target_images]

        images = torch.cat(images, dim=0)
        return images, torch.tensor(leaf_count, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)





class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):  # Reduced patience
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


def train_model(model, train_loader, val_loader, num_epochs, device, learning_rate=1e-4, task='age'):
    """Training function with MAE tracking."""
    if task == 'age':
        criterion = nn.SmoothL1Loss()  # Better for age prediction
    else:
        criterion = nn.MSELoss()  # For leaf count
        
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    early_stopping = EarlyStopping(patience=2, min_delta=0.001)
    
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_mae = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae_sum = 0.0
        train_count = 0
        
        for batch_idx, (images, leaf_counts, ages) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            images = images.to(device)
            
            if task == 'age':
                targets = ages.to(device).float()
            else:
                targets = leaf_counts.to(device).float()
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate MAE for this batch
            with torch.no_grad():
                batch_mae = torch.mean(torch.abs(outputs.squeeze() - targets)).item()
                train_mae_sum += batch_mae
            
            train_loss += loss.item()
            train_count += 1
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for images, leaf_counts, ages in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images = images.to(device)
                
                if task == 'age':
                    targets = ages.to(device).float()
                else:
                    targets = leaf_counts.to(device).float()
                
                outputs, _ = model(images)
                loss = criterion(outputs.squeeze(), targets)
                
                # Calculate MAE for this batch
                batch_mae = torch.mean(torch.abs(outputs.squeeze() - targets)).item()
                val_mae_sum += batch_mae
                
                val_loss += loss.item()
                val_count += 1
        
        # Calculate average losses and MAEs
        avg_train_loss = train_loss / train_count
        avg_val_loss = val_loss / val_count
        avg_train_mae = train_mae_sum / train_count
        avg_val_mae = val_mae_sum / val_count
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train MAE: {avg_train_mae:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val MAE: {avg_val_mae:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Track best MAE for model saving
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            print(f"🎯 New best validation MAE: {best_val_mae:.4f}")
        
        # Early stopping check (based on validation loss)
        if early_stopping(avg_val_loss, model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return model, train_losses, val_losses, train_maes, val_maes

def evaluate_model(model, test_loader, device, task='age'):
    """Evaluation function."""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, leaf_counts, ages in test_loader:
            images = images.to(device)
            
            if task == 'age':
                targets = ages.to(device).float()
            else:
                targets = leaf_counts.to(device).float()
            
            outputs, _ = model(images)
            
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2, y_true, y_pred




# Crop configurations - Using your existing CSV files
CROPS_CONFIG = {
   
    'wheat': {
        'plants': 5,
        'days': 118,
        'root_path': '/Users/mk/Downloads/training_data/',
        'csv_file': '/Users/mk/Downloads/training_data/wheat_train.csv'  # Your actual train CSV
    },
}
    


# Model hyperparameters - Smaller model for faster training
MODEL_CONFIG = {
    'num_images': 4,
    'input_channels': 12,  # 4 images * 3 channels
    'patch_size': 16,
    'num_patches': (224 // 16) ** 2,  # 196
    'projection_dim': 128,  # Reduced from 256 to 128
    'num_heads': 4,         # Reduced from 8 to 4
    'num_layers': 3,        # Reduced from 6 to 3
    'mlp_dim': 256,         # Reduced from 512 to 256
    'dropout_rate': 0.1
}

# Training hyperparameters - Optimized for faster training
TRAINING_CONFIG = {
    'batch_size': 16,  # Increased batch size for faster training
    'num_epochs': 5,   # Drastically reduced for quick training
    'learning_rate': 2e-4,  # Slightly higher LR for faster convergence
    'validation_split': 0.2
}

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================================
# DATA STRUCTURE DEBUGGING AND CSV CREATION UTILITIES
# ============================================================================

def check_data_structure(root_path, crop_name, plants, days):
    """Check if the data structure exists and print the structure."""
    print(f"\nChecking data structure for {crop_name}...")
    print(f"Root path: {root_path}")
    
    crop_path = os.path.join(root_path, crop_name)
    print(f"Crop path: {crop_path}")
    print(f"Crop path exists: {os.path.exists(crop_path)}")
    
    if os.path.exists(crop_path):
        print(f"Contents of {crop_path}:")
        for item in os.listdir(crop_path):
            print(f"  - {item}")
    
    # Check actual plant directories (not just expected ones)
    found_plants = []
    if os.path.exists(crop_path):
        actual_plants = [d for d in os.listdir(crop_path) if d.startswith('p') and os.path.isdir(os.path.join(crop_path, d))]
        actual_plants.sort()
        
        for plant_dir in actual_plants:
            plant_path = os.path.join(crop_path, plant_dir)
            found_plants.append(plant_dir)
            print(f"✅ Found plant {plant_dir}: {plant_path}")
            
            # Check day directories
            actual_days = [d for d in os.listdir(plant_path) if d.startswith('d') and os.path.isdir(os.path.join(plant_path, d))]
            actual_days.sort(key=lambda x: int(x[1:]))
            
            if actual_days:
                print(f"   Days found: {actual_days[:5]}..." if len(actual_days) > 5 else f"   Days found: {actual_days}")
                
                # Check level structure for first found day
                first_day_path = os.path.join(plant_path, actual_days[0])
                levels = ['L1', 'L2', 'L3', 'L4', 'L5']
                found_levels = []
                for level in levels:
                    level_path = os.path.join(first_day_path, level)
                    if os.path.exists(level_path):
                        found_levels.append(level)
                        # Count images in this level
                        images = [f for f in os.listdir(level_path) if f.endswith('.png')]
                        print(f"   Level {level}: {len(images)} images")
                
                print(f"   Levels found: {found_levels}")
            else:
                print(f"   No day directories found in {plant_dir}")
    
    return found_plants

def create_dummy_csv(root_path, crop_name, plants, days, csv_path):
    """Create a dummy CSV file based on actual data structure."""
    if os.path.exists(csv_path):
        print(f"CSV file already exists: {csv_path}")
        return
    
    print(f"Creating dummy CSV file: {csv_path}")
    
    data = []
    levels = ['L1', 'L2', 'L3', 'L4', 'L5']
    
    # Get actual plant and day directories
    crop_path = os.path.join(root_path, crop_name)
    if not os.path.exists(crop_path):
        print(f"Crop directory not found: {crop_path}")
        return
        
    actual_plants = [d for d in os.listdir(crop_path) if d.startswith('p') and os.path.isdir(os.path.join(crop_path, d))]
    actual_plants.sort()
    
    for plant_dir in actual_plants:
        plant_path = os.path.join(crop_path, plant_dir)
        actual_days = [d for d in os.listdir(plant_path) if d.startswith('d') and os.path.isdir(os.path.join(plant_path, d))]
        actual_days.sort(key=lambda x: int(x[1:]))
        
        for day_dir in actual_days:
            day_num = int(day_dir[1:])  # Extract number from 'd12' -> 12
            
            for level in levels:
                level_path = os.path.join(plant_path, day_dir, level)
                if os.path.exists(level_path):
                    # Create filename in the expected format
                    filename = os.path.join(crop_name, plant_dir, day_dir, level, f"{crop_name}_{plant_dir}_{day_dir}_{level}_0.png")
                    
                    # Dummy values - you'll need to replace with actual ground truth
                    leaf_count = min(5 + day_num // 10, 20)  # Dummy leaf count that increases with time
                    age = day_num  # Age is the day number
                    
                    data.append({
                        'filename': filename,
                        'leaf_count': leaf_count,
                        'age': age
                    })
    
    if not data:
        print("No data found to create CSV")
        return
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    df.to_csv(csv_path, index=False)
    print(f"Created dummy CSV with {len(data)} entries")
    print(f"Sample entries:")
    print(df.head())

def debug_and_setup_data():
    """Debug data structure and check existing CSV files."""
    print("="*60)
    print("DATA STRUCTURE DEBUG AND SETUP")
    print("="*60)
    
    for crop_name, config in CROPS_CONFIG.items():
        print(f"\n{'='*40}")
        print(f"Checking {crop_name.upper()}")
        print(f"{'='*40}")
        
        # Check data structure
        found_plants = check_data_structure(
            config['root_path'], 
            crop_name, 
            config['plants'], 
            config['days']
        )
        
        # Check if CSV file exists and show sample
        if os.path.exists(config['csv_file']):
            print(f"✅ CSV file exists: {config['csv_file']}")
            try:
                df = pd.read_csv(config['csv_file'])
                print(f"CSV has {len(df)} entries")
                print("Sample entries:")
                print(df.head())
                print(f"Columns: {list(df.columns)}")
                
                # Check if all required columns exist (case insensitive)
                required_columns = ['filename', 'leaf_count', 'age']
                available_columns_lower = [col.lower() for col in df.columns]
                missing_columns = []
                
                for req_col in required_columns:
                    if req_col.lower() not in available_columns_lower:
                        missing_columns.append(req_col)
                
                if missing_columns:
                    print(f"⚠️  Missing columns: {missing_columns}")
                else:
                    print("✅ All required columns present")
                    # Show the actual column names
                    actual_columns = {req: next(col for col in df.columns if col.lower() == req.lower()) 
                                    for req in required_columns}
                    print(f"Column mapping: {actual_columns}")
                    
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
        else:
            print(f"❌ CSV file not found: {config['csv_file']}")
            print("Please ensure your CSV files are named correctly:")
            print(f"  - {crop_name}_train.csv")
            print("And contain columns: filename, leaf_count, age")

# ============================================================================
# MAIN TRAINING FUNCTION (UPDATED)
# ============================================================================


def train_crop_model(crop_name, task='age'):
    """Train model for a specific crop and task."""
    print(f"\n{'='*60}")
    print(f"Training {crop_name.upper()} - Task: {task.upper()}")
    print(f"{'='*60}")
    
    config = CROPS_CONFIG[crop_name]
    
    # Create dataset using single CSV file
    dataset = CropDataset(
        root_dir=config['root_path'],
        csv_file=config['csv_file'],  # Single CSV file for the crop
        images_per_level=MODEL_CONFIG['num_images'],
        crop=crop_name,
        plants=config['plants'],
        days=config['days'],
        transform=train_transform
    )
    
    print(f"Total dataset size: {len(dataset)} samples")
    
    if len(dataset) == 0:
        print(f"❌ No data found for {crop_name}. Skipping training.")
        return None, [], [], [], []
    
    # For small datasets, use a smaller validation split
    val_ratio = 0.15 if len(dataset) < 100 else 0.2
    train_size = int((1 - val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    
    # Ensure we have at least 1 sample for validation
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
    
    if train_size <= 0:
        print(f"❌ Dataset too small for {crop_name}. Need at least 2 samples.")
        return None, [], [], [], []
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    # Adjust batch size for small datasets
    effective_batch_size = min(TRAINING_CONFIG['batch_size'], train_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=effective_batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 for debugging on macOS
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=effective_batch_size, 
        shuffle=False, 
        num_workers=0,  # Set to 0 for debugging on macOS
        pin_memory=False
    )
    
    # Create model
    model = GromoVisionTransformer(**MODEL_CONFIG)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Adjust training epochs for small datasets
    adjusted_epochs = min(TRAINING_CONFIG['num_epochs'], 8) if len(dataset) < 50 else TRAINING_CONFIG['num_epochs']
    
    # Train model
    model, train_losses, val_losses, train_maes, val_maes = train_model(
        model, train_loader, val_loader, 
        adjusted_epochs, 
        device, 
        TRAINING_CONFIG['learning_rate'],
        task
    )
    
    # Evaluate model
    rmse, mae, r2, _, _ = evaluate_model(model, val_loader, device, task)
    
    print(f"\nFinal Results for {crop_name.upper()} - {task.upper()}:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Best Training MAE: {min(train_maes):.4f}")
    print(f"Best Validation MAE: {min(val_maes):.4f}")
    
    # Save model
    os.makedirs(f'task1' if task == 'age' else 'task2', exist_ok=True)
    model_path = f"task1/{crop_name}_model.pth" if task == 'age' else f"task2/{crop_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model, train_losses, val_losses, train_maes, val_maes



if __name__ == "__main__":
    print("GROMO Challenge 2025 - Training Pipeline")
    print("========================================")
    
    # First, debug and setup data structure
    debug_and_setup_data()
    # Ask user if they want to continue with training
    response = input("\nDo you want to continue with training? (y/n): ").lower()
    if response != 'y':
        print("Training cancelled. Please fix data structure issues and try again.")
        exit()
    
    # Create output directories
    os.makedirs('task1', exist_ok=True)
    os.makedirs('task2', exist_ok=True)
    
    crops = ['wheat']
    tasks = ['age']
    
    # Train all models
    for task in tasks:
        for crop in crops:
            try:
                result = train_crop_model(crop, task)
                if result[0] is not None:  # model is not None
                    model, train_losses, val_losses, train_maes, val_maes = result
                    print(f"✅ Successfully trained {crop} model for {task} prediction")
                    print(f"   Final validation MAE: {val_maes[-1] if val_maes else 'N/A':.4f}")
                else:
                    print(f"⚠️  Skipped {crop} model for {task} prediction (no data)")
            except Exception as e:
                print(f"❌ Error training {crop} model for {task} prediction: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print("\n🎉 Training completed for all crops and tasks!")
    print("📁 Model files saved in task1/ and task2/ directories")
    
    # Plot training curves (optional)
    def plot_training_curves(train_losses, val_losses, title):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training Curves - {title}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'training_curves_{title.lower().replace(" ", "_")}.png')
        plt.show()

