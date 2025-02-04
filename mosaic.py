import numpy as np
import cv2
import gradio as gr
import os
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error



# Step1 :Image Selection and Preprocessing
# Default tiles directory
TILE_DIRECTORY = "tiles"

# Choose a set of test images
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

# Color quantization using k-means clustering
def preprocess_image(image, quantize=True, k=8):
    """
    Preprocess the image (resize and quantize colors).
    Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html 
    
    Inputs:
    image: Input image
    quantize: Whether to quantize colors using k-means clustering
    k: Number of clusters for k-means clustering
    Returns: Preprocessed image
    """
    if quantize:
        h, w, c = image.shape
        
        pixels = image.reshape(-1, 3)
        # Fit k-means clustering, default k=8, run 10 times
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        # Fit and predict the labels
        labels = kmeans.fit_predict(pixels)
        # Get the cluster centers
        new_colors = kmeans.cluster_centers_.astype(np.uint8)
        # Replace the pixel values with the cluster center values
        image = new_colors[labels].reshape(h, w, c)
    return image

# Step 2: Image Grid and Thredsholding
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

# Step 3: Tile Mapping
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

def create_mosaic(image, mapped_tiles, grid_size=16):
    """
    Create a mosaic image by replacing each grid cell with the mapped tile.
    Inputs:
    image: Input image
    mapped_tiles: List of mapped tiles
    grid_size: Grid size for dividing the image, default is 16
    """
    h, w, c = image.shape
    cells_h = h // grid_size
    cells_w = w // grid_size
    mosaic = np.zeros((h, w, c), dtype=np.uint8) # Initialize the mosaic image to (0, 0, 0)
    
    # Loop through each cell and replace with the corresponding tile
    for i in range(cells_h):
        for j in range(cells_w):
            y1, y2 = i * grid_size, (i + 1) * grid_size
            x1, x2 = j * grid_size, (j + 1) * grid_size
            mosaic[y1:y2, x1:x2] = cv2.resize(mapped_tiles[i * cells_w + j], (grid_size, grid_size))
    return mosaic

# Step 5: Similarity Metrics
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


def generate_mosaic(image, grid_size=16):
    """
    Generate a mosaic image using the default tiles directory.
    Inputs:
    image: Input image
    grid_size: Grid size for dividing the image, default is 16
    """
    # Pad the image to fit the grid
    padded_img = padding(image, grid_size)
    
    # Divide the image into grid cells
    grid_colors = divide_image_grid(padded_img, grid_size)
    
    # Load tiles
    tiles, tile_colors = load_tiles(TILE_DIRECTORY)
    
    # Tile mapping
    mapped_tiles = map_tiles(grid_colors, tile_colors, tiles)
    
    # Create the mosaic
    mosaic = create_mosaic(padded_img, mapped_tiles, grid_size)
    
    # Crop the mosaic to the original image size
    mosaic = mosaic[:image.shape[0], :image.shape[1]]
    
    # Compute similarity metrics
    mse, ssim_index = similarity_metrics(image, mosaic)
    
    return mosaic, f"MSE: {mse:.2f}, SSIM: {ssim_index:.2f}"



# Step 4: Building the Gradio Interface
def gradio_interface(image, grid_size=16):
    """
    Gradio interface for generating a mosaic image.
    """
    # Generate the mosaic
    mosaic, metrics = generate_mosaic(image, grid_size)
    return mosaic, metrics

# Create Gradio interface
mosaic_face = gr.Interface(
    fn=gradio_interface,
    # Image and grid size slider
    inputs=[
        gr.Image(type="numpy"),
        gr.Slider(8, 64, step=1, label="Grid Size", value=16)
    ],
    outputs=[
        gr.Image(type="numpy"),
        gr.Textbox(label="Performance Metrics")
    ],
    title="Interactive Image Mosaic Generator",
    description="Upload an image and generate a mosaic. Adjust grid size for mosaic granularity.",
)

# Launch the interface
mosaic_face.launch(share=True)

