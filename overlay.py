import tkinter as tk
import win32gui
import win32con
import win32api
import threading
import time
import keyboard  # New library for the kill switch

from eeg_receiver import EEGReceiver

class StressOverlay:
    def __init__(self):
        # --- SETUP 1: THE BLUE RING (Main Window) ---
        # This window uses "transparent key" transparency. 
        # Anything "black" is invisible. Anything else (Blue Ring) is solid.
        self.root = tk.Tk()
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        
        self.setup_window(self.root)
        self.root.wm_attributes("-transparentcolor", "black")
        self.root.config(bg="black")
        
        # Draw the Blue Ring
        self.canvas = tk.Canvas(self.root, width=self.screen_w, height=self.screen_h, 
                                bg="black", highlightthickness=0)
        self.canvas.pack()
        
        # Inset the rectangle slightly so the thick border is visible
        thickness = 10
        self.canvas.create_rectangle(
            0, 0, self.screen_w, self.screen_h, 
            outline="#00aaff",  # Neon Blue
            width=thickness * 2 # Thickness is centered, so double it to see full width
        )

        # --- SETUP 2: THE RED FLASH (Second Window) ---
        # This window uses "Alpha" transparency for fading.
        # It sits on top of the Blue Ring.
        self.flash_win = tk.Toplevel(self.root)
        self.setup_window(self.flash_win)
        self.flash_win.config(bg="red")
        self.flash_win.wm_attributes("-alpha", 0.0) # Start invisible

        # Status Label (Attached to the Blue Ring window so it's always solid)
        self.label = tk.Label(self.root, text="Initializing...", 
                              font=("Consolas", 14, "bold"), fg="#00aaff", bg="black")
        self.label.place(x=20, y=20)

        # --- CLICK-THROUGH MAGIC ---
        self.root.after(100, lambda: self.set_click_through(self.root, "BlueRingOverlay"))
        self.root.after(100, lambda: self.set_click_through(self.flash_win, "RedFlashOverlay"))

        # --- LOGIC ---
        self.running = True
        
        # Start EEG
        self.eeg = EEGReceiver()
        self.eeg.start()

        # Start Calibration Thread
        self.calibrating = True
        threading.Thread(target=self.run_calibration, daemon=True).start()

        # Start Hotkey Listener (Shift + Esc to Quit)
        keyboard.add_hotkey('shift+esc', self.quit_program)

        # Start Updates
        self.update_overlay()
        self.root.mainloop()

    def setup_window(self, window):
        """Standard setup to make a window fill screen and remove borders"""
        window.overrideredirect(True)
        window.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        window.wm_attributes("-topmost", True)
        window.wm_attributes("-toolwindow", True)

    def set_click_through(self, window, name):
        """Makes a specific window ignore mouse clicks"""
        try:
            # 1. Set the Window Title so we can find it by name
            window.title(name)
            window.update() # Force the name to update immediately

            # 2. Find the window handle (HWND) by that name
            hwnd = win32gui.FindWindow(None, name)
            
            if hwnd == 0:
                print(f"Could not find window: {name}")
                return

            # 3. Apply the 'Transparent' and 'Layered' styles
            styles = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            styles = styles | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, styles)
            
            print(f"Click-through set for: {name}")
        except Exception as e:
            print(f"Error setting click-through: {e}")

    def run_calibration(self):
        # Wait for connection
        while len(self.eeg.AF7Buffer) == 0 and self.running:
            time.sleep(0.1)
        
        # Buffer fill
        time.sleep(3)
        
        if self.running and len(self.eeg.AF7Buffer) > 10:
            self.eeg.find_baseline()
        
        self.calibrating = False

    def update_overlay(self):
        if not self.running:
            return

        # 1. VISUAL UPDATES
        if self.calibrating:
            self.label.config(text="CALIBRATING... (Stay Still)")
        else:
            try:
                tilt = self.eeg.get_tilt_score()
            except:
                tilt = 0.0

            # Update Text
            self.label.config(text=f"TILT: {tilt:.2f} | STOP: Shift+Esc")

            # Update Red Flash (Only visible if stressed)
            if tilt > 0.4:
                intensity = (tilt - 0.4) / 0.6
                alpha = min(max(intensity * 0.6, 0.0), 0.6) # Max 60% opacity
                self.flash_win.wm_attributes("-alpha", alpha)
            else:
                self.flash_win.wm_attributes("-alpha", 0.0)

        # 2. LOOP
        self.root.after(30, self.update_overlay)

    def quit_program(self):
        print("Kill switch activated!")
        self.running = False
        self.root.destroy()
        import sys
        sys.exit()

if __name__ == "__main__":
    StressOverlay()