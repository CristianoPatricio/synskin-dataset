# Dataset Organization

## Directory Structure

```
.
├── data/           # Contains original images and color segmentation masks
└── metadata/       # Contains metadata files
```

## Contents

### Images
- Format: `.jpg`
- Number of images: 10015
- Naming convention: `image_id_synthetic.extension`

### Masks
- Format: Binary masks in `.png`
- One mask per color per image
- Naming convention: `{image_id}_color_{color}.png`

### Metadata
- `annotations_SynSkin_colors.csv`: Contains color image annotations
- `synskin_{train,val,test}_split.csv`: Train/validation/test splits

## File Structure

### annotations_SynSkin_colors.csv
| Column | Description |
|--------|-------------|
| ID | Image ID  |
| light_brown | Binary label (1) is present (0) is absent |
| dark_brown | Binary label (1) is present (0) is absent |
| black | Binary label (1) is present (0) is absent |
| blue-gray | Binary label (1) is present (0) is absent |
| red | Binary label (1) is present (0) is absent |
| white | Binary label (1) is present (0) is absent |

### synskin_{train,val,test}_split.csv
| Column | Description |
|--------|-------------|
| ID | Image ID  |
| light_brown | Binary label (1) is present (0) is absent |
| dark_brown | Binary label (1) is present (0) is absent |
| black | Binary label (1) is present (0) is absent |
| blue-gray | Binary label (1) is present (0) is absent |
| red | Binary label (1) is present (0) is absent |
| white | Binary label (1) is present (0) is absent |