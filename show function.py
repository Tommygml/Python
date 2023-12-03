def show(self):
        #Display the decoded image in a window.
        
        try:
            import tkinter as tk
            from tkinter import ttk
        except ModuleNotFoundError:
            self.show2()
            return
        try:
            from PIL import Image
            from PIL.ImageTk import PhotoImage
        except ModuleNotFoundError:
            print("The Pillow module needs to be installed in order to display the rendered image.")
            print("For installing: https://pillow.readthedocs.io/en/stable/installation.html")

        print("\nRendering the decoded image...")

        # Create the window
        window = tk.Tk()
        window.title(f"Decoded JPEG: {self.file_path.name}")
        try:
            window.state("zoomed")
        except tk.TclError:
            window.state("normal")

        # Horizontal and vertical scrollbars
        scrollbar_h = ttk.Scrollbar(orient = tk.HORIZONTAL)
        scrollbar_v = ttk.Scrollbar(orient = tk.VERTICAL)
        
        # Canvas where the image will be drawn
        canvas = tk.Canvas(
            width = self.image_width,
            height = self.image_height,
            scrollregion = (0, 0, self.image_width, self.image_height),
            xscrollcommand = scrollbar_h.set,
            yscrollcommand = scrollbar_v.set,
        )
        scrollbar_h["command"] = canvas.xview
        scrollbar_v["command"] = canvas.yview

        # Button for saving the image
        save_button = ttk.Button(
            command = self.save,
            text = "Save decoded image",
            padding = 1,
        )
        
        # Convert the image array to a format that Tkinter understands
        my_image = PhotoImage(
            Image.fromarray(
                np.swapaxes(self.image_array, 0, 1)
            )
        )

        # Draw the image to the canvas
        canvas.create_image(0, 0, image=my_image, anchor="nw")

        # Add the canvas and scrollbars to the window
        canvas.pack()
        scrollbar_h.pack(
            side = tk.BOTTOM,
            fill = tk.X,
            before = canvas,
        )
        scrollbar_v.pack(
            side = tk.RIGHT,
            fill = tk.Y,
            before = canvas,
        )

        # Add the save button to the window
        save_button.pack(
            side = tk.TOP,
            before = canvas,
        )

        # Open the window
        window.mainloop()