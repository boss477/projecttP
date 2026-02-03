import os
import cv2
import glob
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Mapping tool names to Class IDs (Modify based on your needs)
TOOL_CLASSES = {
    'Right_Prograsp_Forceps': 0,
    'Left_Prograsp_Forceps': 1,
    'Maryland_Bipolar_Forceps': 2,
    'Cadiere_Forceps': 3,
    # Add others as needed from the ground_truth folder names
}

def convert_dataset(source_dir, dest_dir):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    images_dir = dest_path / 'images'
    labels_dir = dest_path / 'annotations'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frames
    frames_dir = source_path / 'left_frames'
    if not frames_dir.exists():
        print(f"Error: {frames_dir} not found")
        return

    frame_files = sorted(list(frames_dir.glob('*.png')))
    
    print(f"Found {len(frame_files)} frames. processing...")

    for frame_file in tqdm(frame_files):
        frame_name = frame_file.stem
        # Copy image
        shutil.copy(frame_file, images_dir / f"{frame_name}.png")
        
        # Create YOLO label file
        label_file = labels_dir / f"{frame_name}.txt"
        
        yolo_lines = []
        
        # Check masks for each tool
        ground_truth_dir = source_path / 'ground_truth'
        
        img = cv2.imread(str(frame_file))
        h, w = img.shape[:2]
        
        for tool_name_dir in ground_truth_dir.glob('*_labels'):
            tool_raw_name = tool_name_dir.name.replace('_labels', '')
            
            # Map tool name to ID
            # Simple matching, might need refinement based on exact folder names
            class_id = -1
            for key, val in TOOL_CLASSES.items():
                if key in tool_raw_name:
                    class_id = val
                    break
            
            if class_id == -1:
                # print(f"Warning: Unknown tool class for {tool_raw_name}")
                continue
                
            mask_path = tool_name_dir / f"{frame_name}.png"
            if not mask_path.exists():
                continue
                
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or np.sum(mask) == 0:
                continue
                
            # Find bounding box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Normalize for YOLO
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        if yolo_lines:
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
        else:
            # Create empty file if no tools
            open(label_file, 'w').close()

if __name__ == "__main__":
    # Source: Extracted EndoVis zip content
    SOURCE = "data/instrument_dataset_1" 
    # Destination: Ready for training
    DEST = "data/endovis"
    
    convert_dataset(SOURCE, DEST)
