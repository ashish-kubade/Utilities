
import numpy as np
import cv2

def convolve(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Initialize output image
    output = np.zeros_like(image)
    
    # Apply kernel to image
    for i in range(pad_height, image_height - pad_height):
        for j in range(pad_width, image_width - pad_width):
            # Extract the region of interest
            region = image[i - pad_height:i + pad_height + 1, j - pad_width:j + pad_width + 1]
            
            # Apply kernel
            output[i, j] = np.sum(region * kernel)
    
    return output

def gaussian_blur(image, kernel_size, sigma):
    # Create Gaussian kernel
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2)/(2*sigma**2)), (kernel_size, kernel_size))
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    # Apply convolution
    return convolve(image, kernel)

def sobel_filters(image):
    # Sobel filters for gradient computation
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])
    
    # Compute gradients in x and y directions
    gradient_x = convolve(image, sobel_x)
    gradient_y = convolve(image, sobel_y)
    
    return gradient_x, gradient_y

def gradient_magnitude_direction(gradient_x, gradient_y):
    # Compute gradient magnitude and direction
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    
    return magnitude, direction

def non_max_suppression(magnitude, direction):
    # Convert direction to degrees
    direction = np.rad2deg(direction) % 180
    
    # Get image dimensions
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.int32)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Determine neighboring pixels for interpolation
            q, r = 255, 255
            if (0 <= direction[i, j] < 22.5) or (157.5 <= direction[i, j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif (22.5 <= direction[i, j] < 67.5):
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            elif (67.5 <= direction[i, j] < 112.5):
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            elif (112.5 <= direction[i, j] < 157.5):
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            # Perform non-maximum suppression
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
    
    return suppressed

def hysteresis_thresholding(image, low_threshold, high_threshold):
    # Perform hysteresis thresholding
    strong_edge_indices = image >= high_threshold
    weak_edge_indices = (image <= high_threshold) & (image >= low_threshold)
    
    # Initialize array for final edges
    edges = np.zeros_like(image)
    edges[strong_edge_indices] = 255
    
    # Find weak edge pixels connected to strong edge pixels
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            if edges[i, j] == 255:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if edges[i + dx, j + dy] == 128:
                            edges[i + dx, j + dy] = 255
    
    return edges

def canny_edge_detection(image, low_threshold, high_threshold):
    # Apply Gaussian blur
    blurred_image = gaussian_blur(image, kernel_size=5, sigma=1.4)
    
    # Compute gradients
    gradient_x, gradient_y = sobel_filters(blurred_image)
    
    # Compute gradient magnitude and direction
    magnitude, direction = gradient_magnitude_direction(gradient_x, gradient_y)
    
    # Perform non-maximum suppression
    suppressed = non_max_suppression(magnitude, direction)
    
    # Perform hysteresis thresholding
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    
    return edges

# Read image
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Set Canny edge detection parameters
low_threshold = 50
high_threshold = 150

# Perform Canny edge detection
edges = canny_edge_detection(image, low_threshold, high_threshold)

# Display the original and Canny edge detected images
cv2.imshow('Original Image', image)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

