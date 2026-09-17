#!/usr/bin/env python3
"""(Moved to testFiles/) Minimal binary classification sanity check on USPS.

Validates that classes are learnable (e.g., 1 vs 5) independent of anchor framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np

def simple_test():
    """Test if USPS classes 1-4 can be learned with a simple setup"""
    
    # Load USPS from repo root datasets/
    data_root = str(Path(__file__).resolve().parents[2] / 'datasets')
    usps_full = datasets.USPS(root=data_root, train=True, transform=transforms.ToTensor(), download=True)
    
    # Create a simple binary classification: class 1 vs class 5
    # Class 1 has 1005 samples, class 5 has 556 samples
    class_1_indices = [i for i, (_, label) in enumerate(usps_full) if label == 1][:500]
    class_5_indices = [i for i, (_, label) in enumerate(usps_full) if label == 5][:500] 
    
    # Create binary dataset
    class_1_subset = [(usps_full[i][0], 0) for i in class_1_indices]  # Label as 0
    class_5_subset = [(usps_full[i][0], 1) for i in class_5_indices]  # Label as 1
    
    # Combine
    binary_dataset = class_1_subset + class_5_subset
    
    # Simple CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.adapt = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(64 * 4 * 4, 2)  # Binary classification
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = self.adapt(x)
            x = x.flatten(1)
            return self.fc(x)
    
    # Setup training
    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataloader
    loader = DataLoader(binary_dataset, batch_size=32, shuffle=True)
    
    print("Testing USPS class 1 vs class 5 binary classification...")
    print(f"Class 1 (digit 1): {len(class_1_indices)} samples -> label 0")
    print(f"Class 5 (digit 5): {len(class_5_indices)} samples -> label 1")
    
    # Train for a few epochs
    model.train()
    for epoch in range(5):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_y = torch.tensor(batch_y, dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.3f}, Acc={accuracy:.1f}%")
        
    # Test on each class separately
    print("\nTesting on each class:")
    model.eval()
    with torch.no_grad():
        # Test class 1 (should predict label 0)
        class_1_correct = 0
        for i in class_1_indices[:100]:  # Test first 100
            img, _ = usps_full[i]
            output = model(img.unsqueeze(0))
            pred = output.argmax(1).item()
            if pred == 0:
                class_1_correct += 1
        print(f"Class 1 (digit 1) accuracy: {class_1_correct}/100 = {class_1_correct}%")
        
        # Test class 5 (should predict label 1)  
        class_5_correct = 0
        for i in class_5_indices[:100]:
            img, _ = usps_full[i]
            output = model(img.unsqueeze(0))
            pred = output.argmax(1).item()
            if pred == 1:
                class_5_correct += 1
        print(f"Class 5 (digit 5) accuracy: {class_5_correct}/100 = {class_5_correct}%")

if __name__ == "__main__":
    simple_test()