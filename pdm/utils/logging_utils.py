import logging
import os
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import matplotlib.pyplot as plt


def create_image_grid_from_indices(indices, grid_size=(5, 5), image_size=(256, 256), font_size=40):
    # Create a white background image
    grid_width = grid_size[0] * image_size[0]
    grid_height = grid_size[1] * image_size[1]
    background = Image.new('RGB', (grid_width, grid_height), 'white')

    # Create a draw object
    draw = ImageDraw.Draw(background)

    # Use a larger font
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Iterate through indices and place them on the grid
    for i, index in enumerate(indices):
        row = i // grid_size[0]
        col = i % grid_size[0]

        # Calculate the position to place the text
        x = col * image_size[0] + (image_size[0] - font_size) // 2
        y = row * image_size[1] + (image_size[1] - font_size) // 2

        # Draw the index on the image
        draw.text((x, y), str(index), font=font, fill='black')

    # Save or display the resulting image
    return background


def create_heatmap(data, n_rows, n_cols):
    plt.figure()
    data = data.reshape(n_rows, n_cols)
    fig = sns.heatmap(data, cmap='Blues', linewidth=0.5, xticklabels=False, yticklabels=False).get_figure()
    return fig


def init_logging(config):
    config.training.logging.logging_dir = os.path.join(config.training.logging.logging_dir,
                                                       os.getcwd().split('/')[-2],
                                                       config.base_config_path.split('/')[-2],
                                                       config.base_config_path.split('/')[-1].split('.')[0],
                                                       config.wandb_run_name
                                                       )

    os.makedirs(config.training.logging.logging_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
