import numpy as np
from collections import deque, namedtuple
from itertools import product
from math import ceil, cos, pi
from scipy.interpolate import griddata
from typing import Callable, Tuple, Union
from pathlib import Path

# JPEG markers (for our supported segments)
SOI  = bytes.fromhex("FFD8")    # Start of image
SOF0 = bytes.fromhex("FFC0")    # Start of frame (Baseline DCT)
SOF2 = bytes.fromhex("FFC2")    # Start of frame (Progressive DCT)
DHT  = bytes.fromhex("FFC4")    # Define Huffman table
DQT  = bytes.fromhex("FFDB")    # Define quantization table
DRI  = bytes.fromhex("FFDD")    # Define restart interval
SOS  = bytes.fromhex("FFDA")    # Start of scan
DNL  = bytes.fromhex("FFDC")    # Define number of lines
EOI  = bytes.fromhex("FFD9")    # End of image

# Restart markers
MKRS = tuple(bytes.fromhex(hex(marker)[2:]) for marker in range(0xFFD0, 0xFFD8))

# Containers for the parameters of each color component
ColCom = namedtuple("ColCom", "name order vertical_sampling horizontal_sampling quantization_table_id repeat shape")
HuffmanTable = namedtuple("HuffmanTable", "dc ac")

class Mf_JpegDecoder():

    def __init__(self, file:Path) -> None:

        # Open file
        with open(file, "rb") as image:
            self.dt_file = image.read()
        self.file_size = len(self.dt_file)     # Size (in bytes) of the file
        self.file_path = file if isinstance(file, Path) else Path(file)
        
        # Check if file is a JPEG image
        # (The file needs to start with bytes 'FFD8FF')
        """if not self.dt_file.startswith(SOI + b"\xFF"):
            raise NotJpeg("File is not a JPEG image.")
        print(f"Reading file '{file.name}' ({self.file_size:,} bytes)")
        """
        # Handlers for the markers
        self.handlers = {
            DHT: self.define_huffman_table,
            DQT: self.define_quantization_table,
            DRI: self.define_res_intval,
            SOF0: self.st_fme,
            SOF2: self.st_fme,
            SOS: self.start_of_scan,
            EOI: self.end_of_image,
        }

        # Initialize decoding paramenters
        self.file_h = 2            # Offset (in bytes, 0-index) from the beginning of the file
        self.scan_completed = False      # If the 'end of image' marker has been reached
        self.scan_mode = None           # Supported modes: 'baseline_dct' or 'progressive_dct'
        self.img_w = 0            # Width in pixels of the image
        self.img_h = 0           # Height in pixels of the image
        self.col_com = {}      # Hold each color component and its respective paramenters
        self.s_shape = ()          # Size to upsample the subsampled color components
        self.huffman_tables = {}        # Hold all huffman tables
        self.quantization_tables = {}   # Hold all quantization tables
        self.res_intval = 0       # How many MCUs before each restart marker
        self.Img_array = None         # Store the color values for each pixel of the image
        self.sc_number = 0             # Counter for the performed scans

        
        while not self.scan_completed:
            try:
                curdata_byt = self.dt_file[self.file_h]
            except IndexError:
                del self.dt_file
                break

            # Whether the current byte is 0xFF
            if (curdata_byt == 0xFF):

                # Read the next byte
                temp_maker = self.dt_file[self.file_h : self.file_h+2]
                self.file_h += 2

                # Whether the two bytes form a marker (and isn't a restart marker)
                if (temp_maker != b"\xFF\x00") and (temp_maker not in MKRS):

                    # Attempt to get the handler for the marker
                    temp_der = self.handlers.get(temp_maker)
                    temp_SZ = bytes_to_uint(self.dt_file[self.file_h : self.file_h+2]) - 2
                    self.file_h += 2

                    if temp_der is not None:
                        # If a handler was found, pass the control to it
                        my_data = self.dt_file[self.file_h : self.file_h+temp_SZ]
                        temp_der(my_data)
                    else:
                        # Otherwise, just skip the data segment
                        self.file_h += temp_SZ
            
            else:
                # Move to the next byte if the current byte is not 0xFF
                self.file_h += 1

    def st_fme(self, data:bytes) -> None:
        

        dd_size = len(data)
        data_header = 0

        
        
        # Check encoding mode
        # (the marker used for the segment determines the scan mode)
        mode = self.dt_file[self.file_h-4 : self.file_h-2]
        if mode == SOF0:
            self.scan_mode = "baseline_dct"
            print("Scan mode: Sequential")
        elif mode == SOF2:
            #self.scan_mode = "progressive_dct"
            #print("Scan mode: Progressive")
            print("Not support")
        #else:
            #raise UnsupportedJpeg("Encoding mode not supported. Only 'Baseline DCT' and 'Progressive DCT' are supported.")
        
        # Check sample precision
        # (This is the number of bits used to represent each color value of a pixel)
        precision = data[data_header]
        if precision != 8:
            raise UnsupportedJpeg("Unsupported color depth. Only 8-bit greyscale and 24-bit RGB are supported.")
        data_header += 1
        
        # Get image dimensions
        self.img_h = bytes_to_uint(data[data_header : data_header+2])
        data_header += 2
        

        self.img_w = bytes_to_uint(data[data_header : data_header+2])
        data_header += 2
        print(f"Image dimensions: {self.img_w} x {self.img_h}")

        if self.img_w == 0:
            raise CorruptedJpeg("Image width cannot be zero.")

        # Check number of color components
        components_amount = data[data_header]
        if components_amount not in (1, 3):
            if components_amount == 4:
                raise UnsupportedJpeg("CMYK color space is not supported. Only RGB and greyscale are supported.")
            else:
                raise UnsupportedJpeg("Unsupported color space. Only RGB and greyscale are supported.")
        data_header += 1

        if components_amount == 3:
            print("Color space: YCbCr")
        else:
            print("Color space: greyscale")

        # Get the color components and their parameters
        components = (
            "Y",    # Luminance
            "Cb",   # Blue chrominance
            "Cr",   # Red chrominance
        )

        try:
            for count, component in enumerate(components, start=1):
                
                # Get the ID of the color component
                my_id = data[data_header]
                data_header += 1

                # Get the horizontal and vertical sampling of the component
                sample = data[data_header]          # This value is 8 bits long
                horizontal_sample = sample >> 4     # Get the fiMKRS 4 bits of the value
                vertical_sample = sample & 0x0F     # Get the last 4 bits of the value

                data_header += 1

                # Get the quantization table for the component
                my_quantization_table = data[data_header]
                data_header += 1

                # Group the parameters of the component
                my_component = ColCom(
                    name = component,                                       # Name of the color component
                    order = count-1,                                        # Order in which the component will come in the image
                    horizontal_sampling = horizontal_sample,                # Amount of pixels sampled in the horizontal
                    vertical_sampling = vertical_sample,                    # Amount of pixels sampled in the vertical
                    quantization_table_id = my_quantization_table,          # Quantization table selector
                    repeat = horizontal_sample * vertical_sample,           # Amount of times the component repeats during decoding
                    shape = (8*horizontal_sample, 8*vertical_sample),       # Dimensions (in pixels) of the MCU for the component
                )

                # Add the component parameters to the dictionary
                self.col_com.update({my_id: my_component})

                # Have we parsed all components?
                if count == components_amount:
                    break
        
        except IndexError:
            raise CorruptedJpeg("Failed to parse the start of frame.")
        
        # Shape of the sampling area
        # (these values will be used to upsample the subsampled color components)
        sample_width = max(component.shape[0] for component in self.col_com.values())
        sample_height = max(component.shape[1] for component in self.col_com.values())
        self.s_shape = (sample_width, sample_height)

        # Display the samplings
        print(f"Horizontal sampling: {' x '.join(str(component.horizontal_sampling) for component in self.col_com.values())}")
        print(f"Vertical sampling  : {' x '.join(str(component.vertical_sampling) for component in self.col_com.values())}")
        
        # Move the file header to the end of the data segment
        self.file_h += dd_size

    def define_huffman_table(self, data:bytes) -> None:
        """Parse the Huffman tables from the file.
        """
        dd_size = len(data)
        data_header = 0

        

        while (data_header < dd_size):
            table_destination = data[data_header]
            data_header += 1

            # Count how many codes of each length there are
            

            codes_count = {
                bit_length: count
                for bit_length, count
                in zip(range(1, 17), data[data_header : data_header+16])
            }
            data_header += 16

            # Get the Huffman values (HUFFVAL)
            

            huffval_dict = {}   # Dictionary that associates each code bit-length to all its respective Huffman values

            for bit_length, count in codes_count.items():
                huffval_dict.update(
                    {bit_length: data[data_header : data_header+count]}
                )
                data_header += count
            
            # Error checking
            if (data_header > dd_size):
                # If we tried to read more bytes than what the data has, then something is wrong with the file
                raise CorruptedJpeg("Failed to parse Huffman tables.")
            
            # Build the Huffman tree
            

            huffman_tree = {}

            code = 0
            for bit_length, values_list in huffval_dict.items():
                code <<= 1
                for huffval in values_list:
                    code_string = bin(code)[2:].rjust(bit_length, "0")
                    huffman_tree.update({code_string: huffval})
                    code += 1
            
            # Add tree to the Huffman table dictionary
            self.huffman_tables.update({table_destination: huffman_tree})
            print(f"Parsed Huffman table - ", end="")
            print(f"ID: {table_destination & 0x0F} ({'DC' if table_destination >> 4 == 0 else 'AC'})")

           

        # Move the file header to the end of the data segment
        self.file_h += dd_size

    def define_quantization_table(self, data:bytes) -> None:
        """Parse the quantization table from the file.
        """
        dd_size = len(data)
        data_header = 0

        

        # Get all quantization tables on the data
        while (data_header < dd_size):
            table_destination = data[data_header]
            data_header += 1

           

            # Get the 64 values of the 8 x 8 quantization table
            qt_values = np.array([value for value in data[data_header : data_header+64]], dtype="int16")
            try:
                quantization_table = undo_zigzag(qt_values)
            except ValueError:
                raise CorruptedJpeg("Failed to parse quantization tables.")
            data_header += 64

            # Add the table to the quantization tables dictionary
            self.quantization_tables.update({table_destination: quantization_table})
            print(f"Parsed quantization table - ID: {table_destination}")

            
        
        # Move the file header to the end of the data segment
        self.file_h += dd_size

    def define_res_intval(self, data:bytes) -> None:
        """Parse the restart interval value."""
        self.res_intval = bytes_to_uint(data[:2])
        self.file_h += 2
        print(f"Restart interval: {self.res_intval}")
        
        

    def start_of_scan(self, data:bytes) -> None:
        """Parse the information necessary to decode a segment of encoded image data,
        then passes this information to the method that handles the scan mode used."""
        
        dd_size = len(data)
        data_header = 0

        

        # Number of color components in the scan
        components_amount = data[data_header]
        data_header += 1

        # Get parameters of the components in the scan
        my_huffman_tables = {}
        my_col_com = {}
        for component in range(components_amount):
            component_id = data[data_header]    # Should match the component ID's on the 'start of frame'
            data_header += 1

            # Selector for the Huffman tables
            tables = data[data_header]
            data_header += 1
            dc_table =  tables >> 4             # Should match the tables ID's on the 'detect huffman table'
            ac_table = (tables & 0x0F) | 0x10
           

            # Store the parameters
            my_huffman_tables.update({component_id: HuffmanTable(dc=dc_table, ac=ac_table)})
            my_col_com.update({component_id: self.col_com[component_id]})
        
        # Get spectral selection and successive approximation
        if self.scan_mode == "progressive_dct":
            spectral_selection_start = data[data_header]    # Index of the fiMKRS values of the data unit
            spectral_selection_end = data[data_header+1]    # Index of the last values of the data unit
            bit_position_high = data[data_header+2] >> 4    # The position of the last bit sent in the previous scan
            bit_position_low = data[data_header+2] & 0x0F   # The position of the bit sent in the current scan
           
            data_header += 3
        
        # Move the file header to the begining of the entropy encoded segment
        self.file_h += dd_size

        # Define number of lines
        if self.img_h == 0:
            dnl_index = self.dt_file[self.file_h:].find(DNL)
            if dnl_index != -1:
                dnl_index += self.file_h
                self.img_h = bytes_to_uint(self.dt_file[dnl_index+4 : dnl_index+6])
            else:
                raise CorruptedJpeg("Image height cannot be zero.")

        # Dimensions of the MCU (minimum coding unit)
        
        if components_amount > 1:
            self.mcu_width:int = 8 * max(component.horizontal_sampling for component in self.col_com.values())
            self.mcu_height:int = 8 * max(component.vertical_sampling for component in self.col_com.values())
            self.mcu_shape = (self.mcu_width, self.mcu_height)
        else:
            self.mcu_width:int = 8
            self.mcu_height:int = 8
            self.mcu_shape = (8, 8)

        # Amount of MCUs in the whole image (horizontal, vertical, and total)
        
        if components_amount > 1:
            self.mcu_count_h = (self.img_w // self.mcu_width) + (0 if self.img_w % self.mcu_width == 0 else 1)
            self.mcu_count_v = (self.img_h // self.mcu_height) + (0 if self.img_h % self.mcu_height == 0 else 1)
        else:
            component = my_col_com[component_id]
            sample_ratio_h = self.s_shape[0] / component.shape[0]
            sample_ratio_v = self.s_shape[1] / component.shape[1]
            layer_width = self.img_w / sample_ratio_h
            layer_height = self.img_h / sample_ratio_v
            self.mcu_count_h = ceil(layer_width / self.mcu_width)
            self.mcu_count_v = ceil(layer_height / self.mcu_height)
        
        self.mcu_count = self.mcu_count_h * self.mcu_count_v

        # Create the image array (if one does not exist already)
        if self.Img_array is None:
            # 3-dimensional array to store the color values of each pixel on the image
            # array(x-coordinate, y-coordinate, RBG-color)
            count_h = (self.img_w // self.s_shape[0]) + (0 if self.img_w % self.s_shape[0] == 0 else 1)
            count_v = (self.img_h // self.s_shape[1]) + (0 if self.img_h % self.s_shape[1] == 0 else 1)
            self.array_width = self.s_shape[0] * count_h
            self.array_height = self.s_shape[1] * count_v
            self.array_depth = len(self.col_com)
            self.Img_array = np.zeros(shape=(self.array_width, self.array_height, self.array_depth), dtype="int16")
        
        # Setup scan counter
        if self.sc_number == 0:
            self.scan_amount = self.dt_file[self.file_h:].count(SOS) + 1
            print(f"Number of scans: {self.scan_amount}")

        # Begin the scan of the entropy encoded segment
        if self.scan_mode == "baseline_dct":
            self.baseline_dct_scan(my_huffman_tables, my_col_com)
        elif self.scan_mode == "progressive_dct":
            self.progressive_dct_scan(
                my_huffman_tables,
                my_col_com,
                spectral_selection_start,
                spectral_selection_end,
                bit_position_high,
                bit_position_low
            )
        else:
            raise UnsupportedJpeg("Encoding mode not supported. Only 'Baseline DCT' and 'Progressive DCT' are supported.")
    
    def bits_generator(self) -> Callable[[int, bool], str]:
        
        bit_queue = deque()

        # This nested function "remembers" the contents of bit_queue between different calls        
        def get_bits(amount:int=1, restart:bool=False) -> str:
           
            nonlocal bit_queue
            
            # Should be set to 'True' when the restart interval is reached
            if restart:
                bit_queue.clear()       # Discard the remaining bits
                self.file_h += 2   # Jump over the restart marker
            
            # Fetch more bits if the queue has less than the requested amount
            while amount > len(bit_queue):
                next_byte = self.dt_file[self.file_h]
                self.file_h += 1
                
                if next_byte == 0xFF:
                    self.file_h += 1        # Jump over the stuffed byte
                    
                
                bit_queue.extend(
                    np.unpackbits(
                        bytearray((next_byte,))  # Unpack the bits and add them to the end of the queue
                    )
                )
            
            # Return the bits sequence as a string
            return "".join(str(bit_queue.popleft()) for bit in range(amount))
        
        # Return the nested function
        return get_bits

    def baseline_dct_scan(self, huffman_tables_id:dict, my_col_com:dict) -> None:
        
        print(f"\nScan {self.sc_number+1} of {self.scan_amount}")
        print(f"Color components: {', '.join(component.name for component in my_col_com.values())}")
        print(f"MCU count: {self.mcu_count}")
        # print(f"s and performing IDCT...")
        print("In progress...")

        # Function to read the bits from the file's bytes
        next_bits = self.bits_generator()

        # Function to decode the next Huffman value
        def next_huffval() -> int:
            codeword = ""
            huffman_value = None

            while huffman_value is None:
                codeword += next_bits()
                if len(codeword) > 16:
                    raise CorruptedJpeg(f"Failed to decode image ({current_mcu}/{self.mcu_count} MCUs decoded).")
                huffman_value = huffman_table.get(codeword)
            
            return huffman_value

        # Function to perform the inverse discrete cosine transform (IDCT)
        idct = InverseDCT()

        # Function to resize a block of color values
        resize = ResizeGrid()

        # Number of color components in the scan
        components_amount = len(my_col_com)
        
        # Decode all MCUs in the entropy encoded data
        current_mcu = 0
        previous_dc = np.zeros(components_amount, dtype="int16")
        while (current_mcu < self.mcu_count):
           
            
            # (x, y) coordinates, on the image, for the current MCU
            mcu_y, mcu_x = divmod(current_mcu, self.mcu_count_h)
            
            # Loop through all color components
            for depth, (component_id, component) in enumerate(my_col_com.items()):

                # Quantization table of the color component
                quantization_table = self.quantization_tables[component.quantization_table_id]

                # Minimum coding unit (MCU) of the component
                if components_amount > 1:
                    my_mcu = np.zeros(shape=component.shape, dtype="int16")
                    repeat = component.repeat
                else:
                    my_mcu = np.zeros(shape=(8, 8), dtype="int16")
                    repeat = 1

                
                for block_count in range(repeat):
                    # Block of 8 x 8 pixels for the color component
                    block = np.zeros(64, dtype="int16")
                    
                    # DC value of the block
                    table_id = huffman_tables_id[component_id].dc
                    huffman_table:dict = self.huffman_tables[table_id]
                    huffman_value = next_huffval()
                    
                    
                    dc_value = bin_twos_complement(next_bits(huffman_value)) + previous_dc[depth]
                    previous_dc[depth] = dc_value
                    block[0] = dc_value
                    

                    # AC values of the block
                    table_id = huffman_tables_id[component_id].ac
                    huffman_table:dict = self.huffman_tables[table_id]
                    index = 1
                    while index < 64:
                        huffman_value = next_huffval()
                        
                        
                        # A huffman_value of 0 means the 'end of block' (all remaining AC values are zero)
                        if huffman_value == 0x00:
                            break
                        
                        # Amount of zeroes before the next AC value
                        zero_run_length = huffman_value >> 4
                        index += zero_run_length
                        if index >= 64:
                            break

                        # Get the AC value
                        ac_bit_length = huffman_value & 0x0F
                        
                        if ac_bit_length > 0:
                            ac_value = bin_twos_complement(next_bits(ac_bit_length))
                            block[index] = ac_value
                        
                        # Go to the next AC value
                        index += 1
                    
                    # Undo the zigzag scan and apply dequantization
                    block = undo_zigzag(block) * quantization_table

                    # Apply the inverse discrete cosine transform (IDCT)
                    block = idct(block)

                    # Coordinates of the block on the current MCU
                    block_y, block_x = divmod(block_count, component.horizontal_sampling)
                    block_y, block_x = 8*block_y, 8*block_x

                    # Add the block to the MCU
                    my_mcu[block_x : block_x+8, block_y : block_y+8] = block
            
                # Upsample the block if necessary
                if component.shape != self.s_shape:
                    my_mcu = resize(my_mcu, self.s_shape)
                """NOTE
                Linear interpolation is performed on subsampled color components.
                """
                
                # Add the MCU to the image
                x = self.mcu_width * mcu_x
                y = self.mcu_height * mcu_y
                self.Img_array[x : x+self.mcu_width, y : y+self.mcu_height, component.order] = my_mcu
            
            # Go to the next MCU
            current_mcu += 1
            print_progress(current_mcu, self.mcu_count)
            
            # Check for restart interval
            if (self.res_intval > 0) and (current_mcu % self.res_intval == 0) and (current_mcu != self.mcu_count):
                next_bits(amount=0, restart=True)
                previous_dc[:] = 0
            
        self.sc_number += 1
        print_progress(current_mcu, self.mcu_count, done=True)
    
    def end_of_image(self, data:bytes) -> None:
        
        
        # Clip the image array to the image dimensions
        self.Img_array = self.Img_array[0 : self.img_w, 0 : self.img_h, :]
        
        
        # Convert image from YCbCr to RGB
        if (self.array_depth == 3):
            self.Img_array = YCbCr_to_RGB(self.Img_array)
        elif (self.array_depth == 1):
            np.clip(self.Img_array, a_min=0, a_max=255, out=self.Img_array)
            self.Img_array = self.Img_array[..., 0].astype("uint8")
        
        self.scan_completed = True
        self.show()
        del self.dt_file
    
    def show(self):
        #Display the decoded image in a window.
     
    # Check if Pillow is installed, otherwise call show2()
        try:
           from PIL import Image
           from PIL.ImageTk import PhotoImage
        except ModuleNotFoundError:
            print("The Pillow module needs to be installed.")
            self.show2()
            return

        print("\nProceeding to the next step...")
        self.save()
    
    # Here, add any essential operations that need to be performed before proceeding
    # For example, if you need to save the image or perform some processing,
    # include that code here.

    # Example: Call the save function directly (if needed)
    # self.save()
    
    def show2(self):
        #Display the decoded image in the default image viewer of the operating system.
        
        try:
            from PIL import Image
        except ModuleNotFoundError:
            print("The Pillow module needs to be installed in order to display the rendered image.")
            print("For installing it: https://pillow.readthedocs.io/en/stable/installation.html")
            return
        
        img = np.swapaxes(self.Img_array, 0, 1)
        Image.fromarray(img).show()
    
    def save(self) -> None:
        """Open a file dialog to save the image array as an image to the disk.
        """
        from PIL import Image
        from tkinter.filedialog import asksaveasfilename
        print( "Tommy")
        # Open a file dialog for the user to provide a path
        
        img_path = self.file_path.with_suffix('.bmp')
        # If the user has canceled, then exit the function
        if img_path == Path():
            return
        
        # Make sure that the saved image does not overwrite an existing file
        count = 1
        my_stem = img_path.stem
        while img_path.exists():
            img_path = img_path.with_stem(f"{my_stem} ({count})")
            count += 1
        
        # Convert the image array to a PIL object
        my_image = Image.fromarray(np.swapaxes(self.Img_array, 0, 1))

        #  the image to disk
        try:
            my_image.save(img_path)
        except ValueError:
            img_path = img_path.with_suffix(".png")
            count = 1
            my_stem = img_path.stem
            while img_path.exists():
                img_path = img_path.with_stem(f"{my_stem} ({count})")
                count += 1
            my_image.save(img_path, format="png")
        
        print(f"Decoded image was saved to '{img_path}'")


class InverseDCT():
    
    
    # Precalculate the constant values used on the IDCT function
    # (those values are cached, being calculated only the fiMKRS time a instance of the class is created)
    idct_table = np.zeros(shape=(8,8,8,8), dtype="float64")
    xyuv_coordinates = tuple(product(range(8), repeat=4))   # All 4096 combinations of 4 values from 0 to 7 (each)
    xy_coordinates = tuple(product(range(8), repeat=2))     # All 64 combinations of 2 values from 0 to 7 (each)
    for x, y, u, v in xyuv_coordinates:
       
        # Scaling factors
        Cu = 2**(-0.5) if u == 0 else 1.0   # Horizontal
        Cv = 2**(-0.5) if v == 0 else 1.0   # Vertical 

        # Frequency component
        idct_table[x, y, u, v] = 0.25 * Cu * Cv * cos((2*x + 1) * pi * u / 16) * cos((2*y + 1) * pi * v / 16)

    

    def __call__(self, block:np.ndarray) -> np.ndarray:
        
        # Array to store the results
        output = np.zeros(shape=(8, 8), dtype="float64")

        # Summation of the frequecies components
        for x, y in self.xy_coordinates:
            output[x, y] = np.sum(block * self.idct_table[x, y, ...], dtype="float64")
        
        # Return the color values
        return np.round(output).astype(block.dtype) + 128
        

class ResizeGrid():
    

    # Cache the meshes used for the interpolation
    mesh_cache = {}
    indices_cache = {}

    def __call__(self, block:np.ndarray, new_shape:Tuple[int,int]) -> np.ndarray:
        

        # Ratio of the resize
        old_width, old_height = block.shape
        new_width, new_height = new_shape
        key = ((old_width, old_height), (new_width, new_height))

        # Get the interpolation mesh from the cache
        new_xy = self.mesh_cache.get(key)
        if new_xy is None:
            # If the cache misses, then calculate and cache the mesh
            max_x = old_width - 1
            max_y = old_height - 1
            num_points_x = new_width * 1j
            num_points_y = new_height * 1j
            new_x, new_y = np.mgrid[0 : max_x : num_points_x, 0 : max_y : num_points_y]
            new_xy = (new_x, new_y)
            self.mesh_cache.update({key: new_xy})
           
        
        # Get, from the cache, the indices of the values on the original grid
        old_xy = self.indices_cache.get(key[0])
        if old_xy is None:
            # If the cache misses, calculate and cache the indices
            xx, yy = np.indices(block.shape)
            xx, yy = xx.flatten(), yy.flatten()
            old_xy = (xx, yy)
            self.indices_cache.update({key[0]: old_xy})
        
        # Resize the grid and perform linear interpolation
        resized_block = griddata(old_xy, block.ravel(), new_xy)
        
        return np.round(resized_block).astype(block.dtype)


# ----------------------------------------------------------------------------
# Helper functions

def bytes_to_uint(bytes_obj:bytes) -> int:
    """Convert a big-endian sequence of bytes to an unsigned integer."""
    return int.from_bytes(bytes_obj, byteorder="big", signed=False)

def bin_twos_complement(bits:str) -> int:
    """Convert a binary number to a signed integer using the two's complement."""
    if bits == "":
        return 0
    elif bits[0] == "1":
        return int(bits, 2)
    elif bits[0] == "0":
        bit_length = len(bits)
        return int(bits, 2) - (2**bit_length - 1)
    else:
        raise ValueError(f"'{bits}' is not a binary number.")

def undo_zigzag(block:np.ndarray) -> np.ndarray:
    """Takes an 1D array of 64 elements and undo the zig-zag scan of the JPEG
    encoding process. Returns a 2D array (8 x 8) that represents a block of pixels.
    """
    return np.array(
        [[block[0], block[1], block[5], block[6], block[14], block[15], block[27], block[28]],
        [block[2], block[4], block[7], block[13], block[16], block[26], block[29], block[42]],
        [block[3], block[8], block[12], block[17], block[25], block[30], block[41], block[43]],
        [block[9], block[11], block[18], block[24], block[31], block[40], block[44], block[53]],
        [block[10], block[19], block[23], block[32], block[39], block[45], block[52], block[54]],
        [block[20], block[22], block[33], block[38], block[46], block[51], block[55], block[60]],
        [block[21], block[34], block[37], block[47], block[50], block[56], block[59], block[61]],
        [block[35], block[36], block[48], block[49], block[57], block[58], block[62], block[63]]],
        dtype=block.dtype
    ).T # <-- transposes the array
    

# List that undoes the zig-zag ordering for a single element in a band
# (the element index is used on the list, and it returns a (x, y) tuple
# for the coordinates on the data unit)
zagzig = (
    (0, 0), (1, 0), (0, 1), (0, 2), (1, 1), (2, 0), (3, 0), (2, 1),
    (1, 2), (0, 3), (0, 4), (1, 3), (2, 2), (3, 1), (4, 0), (5, 0),
    (4, 1), (3, 2), (2, 3), (1, 4), (0, 5), (0, 6), (1, 5), (2, 4),
    (3, 3), (4, 2), (5, 1), (6, 0), (7, 0), (6, 1), (5, 2), (4, 3),
    (3, 4), (2, 5), (1, 6), (0, 7), (1, 7), (2, 6), (3, 5), (4, 4),
    (5, 3), (6, 2), (7, 1), (7, 2), (6, 3), (5, 4), (4, 5), (3, 6),
    (2, 7), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (7, 4), (6, 5),
    (5, 6), (4, 7), (5, 7), (6, 6), (7, 5), (7, 6), (6, 7), (7, 7)
)

def YCbCr_to_RGB(Img_array:np.ndarray) -> np.ndarray:
    
    print("\nConverting colors from YCbCr to RGB...")
    Y = Img_array[..., 0].astype("float64")
    Cb = Img_array[..., 1].astype("float64")
    Cr = Img_array[..., 2].astype("float64")

    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.34414 * (Cb - 128.0) - 0.71414 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)

    output = np.stack((R, G, B), axis=-1)
    np.clip(output, a_min=0.0, a_max=255.0, out=output)

    return np.round(output).astype("uint8")

def print_progress(current:int, total:int, done:bool=False, header:str="Progress") -> None:
    
    #if not done:
    #   print(f"{header}: {current}/{total} ({current * 100 / total:.2f}%)", end="\r")
    #else:
    #    print(f"{header}: {current}/{total} ({current * 100 / total:.0f}%) DONE!")
    
    if done:
        print("Continuing to the next step...")
# ----------------------------------------------------------------------------
# Decoder exceptions

class JpegError(Exception):
    """Parent of all other exceptions of this decoder."""


class NotJpeg(JpegError):
    """File is not a JPEG image."""

class CorruptedJpeg(JpegError):
    """Failed to parse the file headers."""

class UnsupportedJpeg(JpegError):
    """JPEG image is encoded in a way that our decoder does not support."""


# ----------------------------------------------------------------------------
# Run script

if __name__ == "__main__":
    from sys import argv
    try:
        from tkinter.filedialog import askopenfilename
        import tkinter as tk
        dialog = True
    except ModuleNotFoundError:
        dialog = False

    # Get the JPEG file path
    # If a path was provided as a command line argument, then use it
    if len(argv) > 1:
        jpeg_path = Path(argv[1])
        command = True
    else:
        command = False
    
    while True:
        
        # Open a dialog to ask the user for a image path
        if not command:
            if dialog:
                window = tk.Tk()
                window.state("withdrawn")
                print("Please choose a JPEG image...")
                jpeg_path = Path(
                    askopenfilename(
                        master = None,
                        title = "Decode a JPEG image",
                        filetypes = (
                            ("JPEG images", "*.jpg *.jpeg *.jfif *.jpe *.jif *.jfi"),
                            ("All files", "*.*")
                        )
                    )
                )
                window.destroy()
                
                # Check if the user has chosen something
                # (if the window was closed, then it returns an empty path)
                if jpeg_path == Path():
                    print("No file was selected.")
                    jpeg_path = None
                    break
            
            # If no GUI is available, then use the command prompt to ask the user for a path
            else:
                jpeg_path = Path(input("JPEG path: "))
        
        # Check if the provided path exists
        if jpeg_path.exists():
            break
        
        # Ask the user to try again if the path does not exist
        else:
            command = False
            print(f"File '{jpeg_path.name}' was not found on '{jpeg_path.parent.resolve()}'")
            
            # Ask yes or no
            while True:
                user_input = input("Try again with another file? [y]es / [n]o: ").lstrip().lower()[0]
                if user_input in "yn":
                    break
            
            # Break or continue the "get file path" loop
            if user_input == "y":
                continue
            elif user_input == "n":
                jpeg_path = None
                break

    
    # Decode the image
    if jpeg_path is not None:
        Mf_JpegDecoder(jpeg_path)
    print("Completed!")