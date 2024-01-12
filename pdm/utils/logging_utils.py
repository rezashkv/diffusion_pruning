from PIL import Image, ImageDraw, ImageFont


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
