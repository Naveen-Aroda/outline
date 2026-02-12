#!/usr/bin/env python3
"""
SVG Outline Processor - GUI Application
A modern, user-friendly interface for processing SVG files to extract outlines.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import threading
import sys
import os

# Import the processing functions from process_svg_v2
from process_svg_v2 import process_svg


class SVGProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SVG Outline Processor")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value="output")
        self.processing = False
        
        # Configuration variables
        self.outline_scale = tk.DoubleVar(value=1.4)
        self.epsilon_factor = tk.DoubleVar(value=0.00015)
        self.base_tension = tk.DoubleVar(value=0.6)
        self.angle_threshold = tk.DoubleVar(value=160)
        self.corner_tension_reduction = tk.DoubleVar(value=0.0)
        
        # Setup UI
        self.setup_ui()
        
        # Redirect stdout to text widget
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def setup_ui(self):
        """Create and arrange all UI components"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="SVG Outline Processor", 
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # === Input Section ===
        input_frame = ttk.LabelFrame(main_frame, text="Input Selection", padding="15")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(1, weight=1)
        
        # File/Folder selection buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="📁 Select Folder", 
            command=self.select_folder,
            width=18
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            button_frame, 
            text="📄 Select Single File", 
            command=self.select_file,
            width=18
        ).pack(side=tk.LEFT)
        
        # Selected path display
        ttk.Label(input_frame, text="Selected:").grid(row=1, column=0, sticky=tk.W, pady=5)
        path_entry = ttk.Entry(input_frame, textvariable=self.input_path, state="readonly", width=50)
        path_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        # Output directory
        ttk.Label(input_frame, text="Output Folder:").grid(row=2, column=0, sticky=tk.W, pady=5)
        output_entry = ttk.Entry(input_frame, textvariable=self.output_path, width=50)
        output_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        ttk.Button(
            input_frame, 
            text="Browse", 
            command=self.select_output_folder,
            width=12
        ).grid(row=2, column=2, padx=(10, 0), pady=5)
        
        # === Configuration Section ===
        config_frame = ttk.LabelFrame(main_frame, text="Processing Parameters", padding="15")
        config_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        config_frame.columnconfigure(1, weight=1)
        
        # Outline Scale
        self.create_slider(
            config_frame, 0, 
            "Outline Scale Multiplier:", 
            self.outline_scale, 
            1.0, 2.0, 
            "How much bigger the outline should be (1.0 = same size)"
        )
        
        # Epsilon Factor
        self.create_slider(
            config_frame, 1, 
            "Epsilon Factor:", 
            self.epsilon_factor, 
            0.0001, 0.001, 
            "Point reduction/smoothing (lower = smoother, more points)"
        )
        
        # Base Tension
        self.create_slider(
            config_frame, 2, 
            "Base Tension:", 
            self.base_tension, 
            0.3, 3.0, 
            "Curve intensity (higher = more curved/flowing)"
        )
        
        # Angle Threshold
        self.create_slider(
            config_frame, 3, 
            "Angle Threshold (degrees):", 
            self.angle_threshold, 
            90, 180, 
            "Corner detection threshold (lower = only sharp corners)"
        )
        
        # Corner Tension Reduction
        self.create_slider(
            config_frame, 4, 
            "Corner Smoothing:", 
            self.corner_tension_reduction, 
            0.0, 1.0, 
            "Corner smoothing (0.0 = maximum smoothing, 1.0 = no smoothing)"
        )
        
        # === Control Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(0, 15))
        
        self.process_button = ttk.Button(
            button_frame, 
            text="🚀 Process SVG(s)", 
            command=self.start_processing,
            width=25,
            style="Accent.TButton"
        )
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="🔄 Reset to Defaults", 
            command=self.reset_defaults,
            width=20
        ).pack(side=tk.LEFT, padx=5)
        
        # === Progress Section ===
        progress_frame = ttk.LabelFrame(main_frame, text="Progress & Log", padding="15")
        progress_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var, 
            maximum=100,
            mode='indeterminate'
        )
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status label
        self.status_label = ttk.Label(
            progress_frame, 
            text="Ready to process", 
            font=("Arial", 10)
        )
        self.status_label.grid(row=0, column=1, padx=(10, 0), pady=(0, 10))
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(
            progress_frame, 
            height=12, 
            width=70,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#f5f5f5"
        )
        self.log_text.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for colored output
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("info", foreground="blue")
        
    def create_slider(self, parent, row, label_text, variable, min_val, max_val, tooltip):
        """Create a labeled slider with value display"""
        # Label
        label = ttk.Label(parent, text=label_text, width=25, anchor=tk.W)
        label.grid(row=row, column=0, sticky=tk.W, pady=5)
        
        # Slider
        slider = ttk.Scale(
            parent, 
            from_=min_val, 
            to=max_val, 
            variable=variable,
            orient=tk.HORIZONTAL,
            length=300
        )
        slider.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=(10, 10), pady=5)
        
        # Value display
        value_label = ttk.Label(parent, textvariable=variable, width=10, anchor=tk.E)
        value_label.grid(row=row, column=2, sticky=tk.E, pady=5)
        
        # Tooltip (hover text)
        self.create_tooltip(label, tooltip)
        
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = ttk.Label(
                tooltip, 
                text=text, 
                background="#ffffe0", 
                relief=tk.SOLID, 
                borderwidth=1,
                padding=5,
                font=("Arial", 9)
            )
            label.pack()
            widget.tooltip = tooltip
            
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                
        widget.bind('<Enter>', on_enter)
        widget.bind('<Leave>', on_leave)
        
    def select_folder(self):
        """Open folder selection dialog"""
        folder = filedialog.askdirectory(title="Select Folder with SVG Files")
        if folder:
            self.input_path.set(folder)
            self.log(f"Selected folder: {folder}")
            
    def select_file(self):
        """Open file selection dialog"""
        file = filedialog.askopenfilename(
            title="Select SVG File",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if file:
            self.input_path.set(file)
            self.log(f"Selected file: {file}")
            
    def select_output_folder(self):
        """Open output folder selection dialog"""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path.set(folder)
            
    def reset_defaults(self):
        """Reset all parameters to default values"""
        self.outline_scale.set(1.4)
        self.epsilon_factor.set(0.00015)
        self.base_tension.set(0.6)
        self.angle_threshold.set(160)
        self.corner_tension_reduction.set(0.0)
        self.log("Reset all parameters to default values")
        
    def log(self, message, tag=""):
        """Add a message to the log text area"""
        self.log_text.insert(tk.END, message + "\n", tag)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
    def update_status(self, message):
        """Update the status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def start_processing(self):
        """Start processing in a separate thread"""
        if self.processing:
            messagebox.showwarning("Already Processing", "Please wait for the current process to complete.")
            return
            
        input_path = self.input_path.get().strip()
        if not input_path:
            messagebox.showerror("No Input", "Please select a folder or file to process.")
            return
            
        if not os.path.exists(input_path):
            messagebox.showerror("Invalid Path", f"The selected path does not exist:\n{input_path}")
            return
            
        # Start processing in background thread
        self.processing = True
        self.process_button.config(state="disabled", text="Processing...")
        self.progress_bar.start()
        self.log_text.delete(1.0, tk.END)
        
        thread = threading.Thread(target=self.process_files, daemon=True)
        thread.start()
        
    def process_files(self):
        """Process the selected files"""
        try:
            input_path = self.input_path.get().strip()
            output_dir = self.output_path.get().strip() or "output"
            
            # Get configuration values
            outline_scale = self.outline_scale.get()
            epsilon_factor = self.epsilon_factor.get()
            base_tension = self.base_tension.get()
            angle_threshold = self.angle_threshold.get()
            corner_tension_reduction = self.corner_tension_reduction.get()
            
            self.log("=" * 60, "info")
            self.log("SVG Outline Processor - Starting", "info")
            self.log("=" * 60, "info")
            self.log(f"Input: {input_path}")
            self.log(f"Output: {output_dir}")
            self.log(f"Configuration:")
            self.log(f"  - Outline scale: {outline_scale}x")
            self.log(f"  - Epsilon factor: {epsilon_factor}")
            self.log(f"  - Base tension: {base_tension}")
            self.log(f"  - Angle threshold: {angle_threshold}°")
            self.log(f"  - Corner smoothing: {corner_tension_reduction}")
            self.log("")
            
            # Determine if input is a file or folder
            if os.path.isfile(input_path):
                # Single file
                if not input_path.lower().endswith('.svg'):
                    self.log(f"ERROR: {input_path} is not an SVG file", "error")
                    self.update_status("Error: Not an SVG file")
                    return
                    
                self.update_status(f"Processing: {os.path.basename(input_path)}")
                self.log(f"Processing single file: {input_path}", "info")
                
                result = process_svg(
                    input_path,
                    output_dir,
                    outline_scale,
                    angle_threshold,
                    corner_tension_reduction,
                    epsilon_factor,
                    base_tension
                )
                
                if result:
                    self.log(f"✓ Successfully processed: {os.path.basename(input_path)}", "success")
                    self.update_status("Processing complete!")
                else:
                    self.log(f"✗ Failed to process: {os.path.basename(input_path)}", "error")
                    self.update_status("Processing failed")
                    
            else:
                # Folder - process all SVG files
                svg_files = list(Path(input_path).glob("*.svg"))
                
                if not svg_files:
                    self.log(f"ERROR: No SVG files found in {input_path}", "error")
                    self.update_status("Error: No SVG files found")
                    return
                    
                self.log(f"Found {len(svg_files)} SVG file(s) to process", "info")
                self.log("")
                
                success_count = 0
                for i, svg_file in enumerate(svg_files, 1):
                    self.update_status(f"Processing {i}/{len(svg_files)}: {svg_file.name}")
                    self.log(f"[{i}/{len(svg_files)}] Processing: {svg_file.name}", "info")
                    
                    try:
                        result = process_svg(
                            str(svg_file),
                            output_dir,
                            outline_scale,
                            angle_threshold,
                            corner_tension_reduction,
                            epsilon_factor,
                            base_tension
                        )
                        
                        if result:
                            success_count += 1
                            self.log(f"  ✓ Success", "success")
                        else:
                            self.log(f"  ✗ Failed", "error")
                            
                    except Exception as e:
                        self.log(f"  ✗ Error: {str(e)}", "error")
                        import traceback
                        self.log(traceback.format_exc(), "error")
                    
                    self.log("")
                
                self.log("=" * 60, "info")
                self.log(f"Completed: {success_count}/{len(svg_files)} files processed successfully", "info")
                self.log("=" * 60, "info")
                self.update_status(f"Complete: {success_count}/{len(svg_files)} successful")
                
        except Exception as e:
            self.log(f"ERROR: {str(e)}", "error")
            self.update_status("Error occurred")
            import traceback
            self.log(traceback.format_exc(), "error")
            
        finally:
            # Reset UI state
            self.processing = False
            self.progress_bar.stop()
            self.process_button.config(state="normal", text="🚀 Process SVG(s)")
            self.update_status("Ready to process")


def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Try to set a modern theme if available
    try:
        style = ttk.Style()
        # Try to use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
    except:
        pass
    
    app = SVGProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()



