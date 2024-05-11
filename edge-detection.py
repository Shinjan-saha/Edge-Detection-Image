from PIL import Image

def grayscale(image):
    return image.convert("L")

def sobel_operator(image):
    # Define Sobel kernels
    kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

    width, height = image.size
    sobel_x = [[0] * height for _ in range(width)]
    sobel_y = [[0] * height for _ in range(width)]

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            pixel_values = [
                image.getpixel((x - 1, y - 1)),
                image.getpixel((x, y - 1)),
                image.getpixel((x + 1, y - 1)),
                image.getpixel((x - 1, y)),
                image.getpixel((x, y)),
                image.getpixel((x + 1, y)),
                image.getpixel((x - 1, y + 1)),
                image.getpixel((x, y + 1)),
                image.getpixel((x + 1, y + 1))
            ]

            # Calculate the gradient in x and y directions
            gradient_x = sum([kernel_x[i][j] * pixel_values[i * 3 + j] for i in range(3) for j in range(3)])
            gradient_y = sum([kernel_y[i][j] * pixel_values[i * 3 + j] for i in range(3) for j in range(3)])

            sobel_x[x][y] = gradient_x
            sobel_y[x][y] = gradient_y

    return sobel_x, sobel_y

def combine_gradients(gradient_x, gradient_y, threshold):
    width, height = len(gradient_x), len(gradient_x[0])
    combined = Image.new("L", (width, height))

    for x in range(width):
        for y in range(height):
            magnitude = (gradient_x[x][y]**2 + gradient_y[x][y]**2)**0.5
            combined.putpixel((x, y), 255 if magnitude > threshold else 0)

    return combined

if __name__ == "__main__":
    # Load the image
    original_image = Image.open("apple.jpg")

    # Convert the image to grayscale
    gray_image = grayscale(original_image)

    # Apply Sobel operator to get gradients in x and y directions
    sobel_x, sobel_y = sobel_operator(gray_image)

    # Combine gradients and apply threshold for edge detection
    threshold = 100  # Adjust this threshold as needed
    edge_image = combine_gradients(sobel_x, sobel_y, threshold)

    # Save the edge-detected image
    edge_image.save("edge_detected_image.jpg")
