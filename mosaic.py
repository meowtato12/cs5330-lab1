import numpy as np
import cv2
import gradio as gr
import os
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


"""mosaic.ipynb

## 1. Setup and Utilities
"""
TILE_DIRECTORY = "tiles"


print("📚 Libraries imported successfully!")

"""## 2. Image Processing"""

# Load tile images from directory
def load_tiles(tile_directory):
    """
    Load tiles from the specified directory and compute their average colors.
    Returns a list of tiles and a list of average colors.
    input: tile_directory
    """
    tiles = []
    tile_colors = []
    for filename in os.listdir(tile_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            tile = cv2.imread(os.path.join(tile_directory, filename))
            if tile is not None:
                # Convert BGR to RGB
                tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                tiles.append(tile)
                # Compute the average color of the tile
                avg_color = np.mean(tile, axis=(0, 1)).astype(np.uint8)
                # Get the RGB tuple
                tile_colors.append(tuple(avg_color))
    return tiles, tile_colors

# Image Grid and Thredsholding
def divide_image_grid(image, grid_size=16):
    """
    Divide the image into a grid and compute the average color of each cell.
    Input:
    image: Input image
    grid_size: Grid size for dividing the image, default is 16
    Returns: List of average colors for each grid cell
    """
    # Get the image dimensions
    h, w, c = image.shape
    # Calculate the number of cells in the grid
    cells_h = h // grid_size
    cells_w = w // grid_size
    grid_colors = []

    # Loop through each cell to compute the average color
    for i in range(cells_h):
        for j in range(cells_w):
            y1, y2 = i * grid_size, (i + 1) * grid_size
            x1, x2 = j * grid_size, (j + 1) * grid_size
            cell = image[y1:y2, x1:x2]
            avg_color = np.mean(cell, axis=(0, 1)).astype(np.uint8)
            grid_colors.append(tuple(avg_color))
    return grid_colors

# Deal with not divisible grid size
def padding(image, grid_size):
    """
    Pad the image so that its dimensions are divisible by the grid size.
    Input:
    image: Input image
    grid_size: Grid size for dividing the image
    Returns: Padded image
    """
    # Get the remainders
    h, w, _ = image.shape
    pad_h = (grid_size - h % grid_size) % grid_size
    pad_w = (grid_size - w % grid_size) % grid_size

    # Pad the image (using edge pixels for padding)
    padded_img = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded_img

MAX_IMAGE_SIZE = 512  # Reduce image size for faster processing

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

"""## 3. Performance Metrics"""

# Similarity Metrics
def similarity_metrics(original, mosaic):
    """
    Compute the similarity between the original image and the mosaic.
    Inputs:
    original: Original image
    mosaic: Mosaic image
    Returns: Mean squared error (MSE) and structural similarity index (SSIM)

    Resources:
    MSE: Lower MSE indicates higher similarity
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.mean_squared_error
    SSIM: Structural similarity index, higher values indicate higher similarity
    https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
    """
    # Compute the mean squared error (MSE)
    # mse = mean_squared_error(original, mosaic)

    mse = np.mean((original - mosaic) ** 2)

    # Find the minimum dimension for the window size
    min_dimension = min(original.shape[0], original.shape[1], mosaic.shape[0], mosaic.shape[1])

    # Use 7 or the smallest dimension
    win_size = min(7, min_dimension)

    # Ensure win_size is odd to find the center pixel
    if win_size % 2 == 0:
        win_size -= 1

    # Compute the structural similarity index (SSIM)
    ssim_index = ssim(original, mosaic, win_size=win_size, channel_axis=-1)
    return mse, ssim_index

# Tile Mapping
def map_tiles(grid_colors, tile_colors, tiles):
    """
    Map each grid cell to the closest tile based on color.
    Inputs:
    grid_colors: List of average colors for each grid cell
    tile_colors: List of average colors for each tile
    tiles: List of tile images
    Returns: List of mapped tiles
    Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    """
    # Use KDTree for fast nearest-neighbor lookup
    tree = KDTree(tile_colors)
    mapped_tiles = []
    # Find the closest tile for each grid cell and append to the list
    for color in grid_colors:
        _, idx = tree.query(color)
        mapped_tiles.append(tiles[idx])
    return mapped_tiles

"""## 4. Mosaic File Handling"""

def create_mosaic(image, mapped_tiles, grid_colors, grid_size=16, color_only=False):
    """
    Create a mosaic image either using tiles or solid colors.
    """
    h, w, c = image.shape
    cells_h = h // grid_size
    cells_w = w // grid_size
    mosaic = np.zeros((h, w, c), dtype=np.uint8)  # Initialize blank mosaic

    for i in range(cells_h):
        for j in range(cells_w):
            y1, y2 = i * grid_size, (i + 1) * grid_size
            x1, x2 = j * grid_size, (j + 1) * grid_size
            if color_only:
                # Fill grid cell with solid color instead of tile
                mosaic[y1:y2, x1:x2] = np.full((grid_size, grid_size, 3), grid_colors[i * cells_w + j], dtype=np.uint8)
            else:
                # Replace with closest tile
                mosaic[y1:y2, x1:x2] = cv2.resize(mapped_tiles[i * cells_w + j], (grid_size, grid_size))

    return mosaic

def generate_mosaic(image, grid_size=16, color_only=False):
    tiles, tile_colors = load_tiles(TILE_DIRECTORY)  # Use correct directory
    if not tiles:
        return None, "Error: No tiles found in the directory!"

    grid_colors = divide_image_grid(image, grid_size)
    mapped_tiles = map_tiles(grid_colors, tile_colors, tiles)

    # Pass color_only to create_mosaic
    mosaic = create_mosaic(image, mapped_tiles, grid_colors, grid_size, color_only=color_only)

    # Compute similarity metrics
    mse, ssim_index = similarity_metrics(image, mosaic)
    metrics = f"MSE: {mse:.2f}, SSIM: {ssim_index:.4f}"

    return mosaic, metrics


def save_selected_format(mosaic, file_format):
    return save_mosaic(mosaic, file_format)

def save_mosaic(mosaic, file_format="png"):
    save_path = f"mosaic.{file_format}"  # Save in the same directory as mosaic.py
    if file_format == "jpeg":
        cv2.imwrite(save_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    else:
        cv2.imwrite(save_path, cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR))
    return save_path

"""## 5. Gradio Interface"""

# Gradio Interface function
def gradio_interface(image, grid_size=16, file_format="png", color_only=False):
    """
    Generates a mosaic from the uploaded image and saves it in the selected format.
    """
    # Resize the image to a manageable size
    resized_image = resize_image(image)
    # Pad the image to be divisible by the grid size
    padded_image = padding(resized_image, grid_size)
    # Generate the mosaic using the padded image
    mosaic, metrics = generate_mosaic(padded_image, grid_size, color_only)
    if mosaic is None:
        return None, "Error: Mosaic generation failed! Check tile images.", None

    # Save the mosaic image in the selected format
    save_path = save_mosaic(mosaic, file_format)

    return mosaic, metrics, save_path

# Gradio UI
mosaic_face = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),
        gr.Slider(8, 64, step=8, label="Grid Size (Tile Size)", value=16),
        gr.Radio(["png", "jpeg"], label="Download Format", value="png"),
        gr.Checkbox(label="Color Only Mode")  # <-- Add this
    ],
    outputs=[
        gr.Image(type="numpy", label="Generated Mosaic"),
        gr.Textbox(label="Performance Metrics (MSE & SSIM)"),
        gr.File(label="Download Mosaic File")
    ],
    title="🎨 Interactive Image Mosaic Generator",
    description=(
        "📌 **How it works:**\n"
        "1️⃣ Upload an image 📷\n"
        "2️⃣ Adjust grid size (smaller = more detail) 🔳\n"
        "3️⃣ Choose format (PNG/JPEG) 💾\n"
        "4️⃣ Enable 'Color Only Mode' for a simplified version 🎨\n"
        "5️⃣ Download your mosaic! 🎉"
    ),
    theme="compact",
)

"""## Excution"""

if __name__ == "__main__":
    mosaic_face.launch(debug=True)
