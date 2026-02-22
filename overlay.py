import tkinter as tk
import win32gui
import win32con
import win32api
import threading
import time
import keyboard 
import random

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
        thickness = 8
        self.canvas.create_rectangle(
            0, 0, self.screen_w, self.screen_h, 
            outline="#00aaff",  # Neon Blue
            width=thickness * 2 # Thickness is centered, so double it to see full width
        )

        # --- PROGRESS BAR UI (Hidden by default) ---
        self.cx = self.screen_w // 2
        self.cy = self.screen_h // 2
        self.bar_w = 500
        self.bar_h = 40
        
        # Background outline of the bar
        self.prog_bg = self.canvas.create_rectangle(
            self.cx - self.bar_w//2, self.cy - self.bar_h//2, 
            self.cx + self.bar_w//2, self.cy + self.bar_h//2, 
            outline="#00aaff", width=3, state="hidden"
        )
        # The fill of the bar
        self.prog_fill = self.canvas.create_rectangle(
            self.cx - self.bar_w//2, self.cy - self.bar_h//2, 
            self.cx - self.bar_w//2, self.cy + self.bar_h//2, 
            fill="#00aaff", state="hidden"
        )
        # The Text above the bar
        self.prog_text = self.canvas.create_text(
            self.cx, self.cy - 40, text="CALIBRATING BIOMETRICS... 0%", 
            fill="#00aaff", font=("Consolas", 20, "bold"), state="hidden"
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
        self.calibrating = True
        self.calibration_start_time = time.time()
        
        self.shaking = False
        self.shake_hwnd = None
        self.shake_orig_rect = None

        #Smoothing tilt
        self.previous_tilt = 0.0
        self.smoothing_factor = 0.1
        
        # Start EEG
        self.eeg = EEGReceiver()
        self.eeg.start()

        # Start Hotkey Listener (Shift + Esc to Quit)
        keyboard.add_hotkey('shift+esc', self.quit_program)
        keyboard.add_hotkey('ctrl+shift+r', self.trigger_recalibration)

        # Start Updates
        threading.Thread(target=self.run_calibration, daemon=True).start()
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

    def trigger_recalibration(self):
        """Called when user presses Shift + `"""
        if not self.calibrating:
            print("Manual Recalibration Triggered!")
            
            # 1. Snap the window back if it was currently shaking!
            if self.shaking and self.shake_hwnd and self.shake_orig_rect:
                x, y, r, b = self.shake_orig_rect
                win32gui.SetWindowPos(
                    self.shake_hwnd, 0, x, y, 0, 0, 
                    win32con.SWP_NOZORDER | win32con.SWP_NOSIZE
                )
                self.shaking = False
                self.shake_hwnd = None
                self.shake_orig_rect = None
            
            # 2. Hide the red flash immediately
            self.flash_win.wm_attributes("-alpha", 0.0)
            
            # 3. Start calibration process again
            self.calibrating = True
            threading.Thread(target=self.run_calibration, daemon=True).start()

    def run_calibration(self):
        # Wait for connection
        while len(self.eeg.AF7Buffer) == 0 and self.running:
            print("Waiting for connection")
            time.sleep(0.1)
        
        self.calibration_start_time = time.time()
        # Buffer fill
        time.sleep(3)
        
        if self.running and len(self.eeg.AF7Buffer) > 10:
            self.eeg.find_baseline()
        
        self.calibrating = False

    def update_overlay(self):
        if not self.running:
            return

        #Forcing the window to stay on top
        self.root.wm_attributes("-topmost", True)
        self.flash_win.wm_attributes("-topmost", True)

        # 1. VISUAL UPDATES
        if self.calibrating:
            self.label.config(text="CALIBRATING... (Stay Still)")

            # Show the Loading Bar UI
            self.canvas.itemconfig(self.prog_bg, state="normal")
            self.canvas.itemconfig(self.prog_fill, state="normal")
            self.canvas.itemconfig(self.prog_text, state="normal")
            
            # Calculate progress (3s buffer + 30s baseline = 33 seconds total)
            elapsed = time.time() - self.calibration_start_time
            progress = min(elapsed / 33.0, 1.0)
            
            # Update the text percentage
            self.canvas.itemconfig(self.prog_text, text=f"CALIBRATING BIOMETRICS... {int(progress * 100)}%")
            
            # Update the width of the filled bar
            fill_x2 = (self.cx - self.bar_w//2) + (self.bar_w * progress)
            self.canvas.coords(self.prog_fill, 
                               self.cx - self.bar_w//2, self.cy - self.bar_h//2, 
                               fill_x2, self.cy + self.bar_h//2)
        else:
            # Hide the Loading Bar UI
            self.canvas.itemconfig(self.prog_bg, state="hidden")
            self.canvas.itemconfig(self.prog_fill, state="hidden")
            self.canvas.itemconfig(self.prog_text, state="hidden")

            try:
                tilt = self.eeg.get_tilt_score()
                if self.previous_tilt is not None:
                    tilt = ((1 - self.smoothing_factor) * tilt) + (self.smoothing_factor * self.previous_tilt)
                self.previous_tilt = tilt
            except:
                tilt = 0.0

            # Update Text
            self.label.config(text=f"TILT: {tilt:.2f} | STOP: Shift+Esc | RECALIB: Ctrl+Shift+R")

            # Update Red Flash (Only visible if stressed)
            if tilt > 0.4:
                intensity = (tilt - 0.4) / 0.6
                alpha = min(max(intensity * 0.6, 0.0), 0.6) # Max 60% opacity
                self.flash_win.wm_attributes("-alpha", alpha)
            else:
                self.flash_win.wm_attributes("-alpha", 0.0)
            
            if tilt > 0.7:
                if not self.shaking:
                    # Just crossed the threshold! Find the active game window.
                    active_window = win32gui.GetForegroundWindow()
                    
                    # Make sure we don't accidentally shake our own overlay or the desktop
                    if active_window and active_window != int(self.root.frame(), 16):
                        self.shake_hwnd = active_window
                        # Save the original position so we can snap it back later
                        self.shake_orig_rect = win32gui.GetWindowRect(self.shake_hwnd)
                        self.shaking = True

                if self.shaking and self.shake_hwnd:
                    # We are currently shaking. Generate random X/Y offsets.
                    x, y, r, b = self.shake_orig_rect
                    
                    # Maximum shake distance in pixels (adjust these for more/less chaos)
                    dx = random.randint(-15, 15)
                    dy = random.randint(-15, 15)
                    
                    # Move the window
                    win32gui.SetWindowPos(
                        self.shake_hwnd, 
                        0, 
                        x + dx, y + dy, 
                        0, 0, # Width and height are ignored due to SWP_NOSIZE
                        win32con.SWP_NOZORDER | win32con.SWP_NOSIZE
                    )
            else:
                # Tilt is under 0.7. Snap the window back to normal if it was shaking.
                if self.shaking:
                    if self.shake_hwnd and self.shake_orig_rect:
                        x, y, r, b = self.shake_orig_rect
                        win32gui.SetWindowPos(
                            self.shake_hwnd, 
                            0, 
                            x, y, 
                            0, 0, 
                            win32con.SWP_NOZORDER | win32con.SWP_NOSIZE
                        )
                    self.shaking = False
                    self.shake_hwnd = None
                    self.shake_orig_rect = None

        # 2. LOOP
        self.root.after(30, self.update_overlay)

    def quit_program(self):
        print("Kill switch activated!")
        
        # Snap the game back to normal if we quit while shaking
        if self.shaking and self.shake_hwnd and self.shake_orig_rect:
            x, y, r, b = self.shake_orig_rect
            win32gui.SetWindowPos(
                self.shake_hwnd, 0, x, y, 0, 0, 
                win32con.SWP_NOZORDER | win32con.SWP_NOSIZE
            )

        self.running = False
        self.root.destroy()
        import sys
        sys.exit()

if __name__ == "__main__":
    StressOverlay()