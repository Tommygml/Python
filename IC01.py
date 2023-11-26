# [The existing code for the JPEG class and other supporting classes/functions goes here]

def write_to_bmp(filename, width, height, pixel_data):
    with open(filename, 'wb') as f:
        # BMP Header
        f.write(b'BM')  # Signature
        file_size = 54 + 3 * width * height
        f.write(struct.pack('<I', file_size))  # File size
        f.write(b'\x00\x00')  # Reserved
        f.write(b'\x00\x00')  # Reserved
        f.write(struct.pack('<I', 54))  # Offset to start of Pixel Data
        f.write(struct.pack('<I', 40))  # Header Size
        f.write(struct.pack('<I', width))  # Image width
        f.write(struct.pack('<I', height))  # Image height
        f.write(struct.pack('<H', 1))  # Planes
        f.write(struct.pack('<H', 24))  # Bits per Pixel
        f.write(struct.pack('<I', 0))  # Compression
        f.write(struct.pack('<I', 0))  # Image size (no compression)
        f.write(struct.pack('<I', 0))  # X Pixels per meter
        f.write(struct.pack('<I', 0))  # Y Pixels per meter
        f.write(struct.pack('<I', 0))  # Total Colors
        f.write(struct.pack('<I', 0))  # Important Colors

        # Pixel Data
        for y in range(height):
            for x in range(width):
                f.write(pixel_data[y][x])

# Modified JPEG class to include BMP writing
class JPEG:
    # [Existing methods of the JPEG class]
    
    def decode_to_bmp(self, bmp_filename):
        self.decode()
        # Assume self.pixel_data contains the decoded pixel data
        # The structure of self.pixel_data should be a 2D array of RGB tuples
        write_to_bmp(bmp_filename, self.width, self.height, self.pixel_data)

if __name__ == "__main__":
    jpeg_filename = "input.jpg"
    bmp_filename = "output.bmp"
    img = JPEG(jpeg_filename)
    img.decode_to_bmp(bmp_filename)
