#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Xbox Controller Test Program
"""

import time
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter
from u8_xbox import XboxController

def main():
    # Load a simple MuJoCo model (using the same XML as in your main program)
    model = mujoco.MjModel.from_xml_path("urdf/0726scene3.xml")
    data = mujoco.MjData(model)
    
    # Initialize viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Create Xbox controller instance
        xbox = XboxController(model, data, viewer)
        
        # Create rate limiter for consistent update rate
        rate = RateLimiter(10)  # 10 Hz update rate (slower for readability)
        
        print("Xbox Controller Test Program")
        print("Press Ctrl+C to exit")
        print("-" * 50)
        
        try:
            while viewer.is_running():
                # Update the controller state
                xbox.update()
                
                # Print available attributes
                attrs = [attr for attr in dir(xbox) if not attr.startswith('_') and not callable(getattr(xbox, attr))]
                print("Available attributes:", [attr for attr in attrs if attr not in ['model', 'data', 'viewer']])
                
                # Try common joystick attribute names
                joystick_attrs = []
                for attr in ['left_x', 'left_y', 'right_x', 'right_y', 'left_trigger', 'right_trigger', 
                            'LeftX', 'LeftY', 'RightX', 'RightY', 'LeftTrigger', 'RightTrigger']:
                    if hasattr(xbox, attr):
                        joystick_attrs.append(f"{attr}: {getattr(xbox, attr):.3f}")
                
                if joystick_attrs:
                    print("Joystick values:", " | ".join(joystick_attrs))
                else:
                    print("No joystick attributes found")
                
                # Check button states (try common names)
                buttons = []
                button_names = ['A', 'B', 'X', 'Y', 'LB', 'RB', 'back', 'start', 
                               'LeftStick', 'RightStick', 'a', 'b', 'x', 'y']
                for btn in button_names:
                    if hasattr(xbox, btn) and getattr(xbox, btn):
                        buttons.append(btn)
                
                if buttons:
                    print(f"Buttons pressed: {', '.join(buttons)}")
                
                # Print control state if available
                if hasattr(xbox, 'get_control_state'):
                    try:
                        control_state, manual_mode = xbox.get_control_state()
                        print(f"Control State: {control_state} | Manual Mode: {manual_mode}")
                    except:
                        pass
                
                print("-" * 50)
                
                # Maintain update rate
                rate.sleep()
                
        except KeyboardInterrupt:
            print("\nExiting program...")
        except Exception as e:
            print(f"Error occurred: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()