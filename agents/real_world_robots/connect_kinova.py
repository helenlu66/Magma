#!/usr/bin/env python3
"""
Kinova Gen3 Arm Connection Script
Connects to and controls a Kinova Gen3 robotic arm using the Kortex API.
"""

import sys
import os
import time
import threading

# Set protobuf implementation to pure Python for compatibility
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.SessionClientRpc import SessionClient
from kortex_api.autogen.messages import Base_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient
from kortex_api.TCPTransport import TCPTransport

class KinovaGen3Controller:
    def __init__(self, ip_address="192.168.1.10", username="admin", password="admin"):
        """
        Initialize Kinova Gen3 controller
        
        Args:
            ip_address (str): IP address of the Kinova arm
            username (str): Username for authentication
            password (str): Password for authentication
        """
        self.ip_address = ip_address
        self.username = username
        self.password = password
        
        # Connection objects
        self.transport = None
        self.router = None
        self.session_client = None
        self.base_client = None
        
        # Connection status
        self.is_connected = False

    def connect(self):
        """
        Establish connection to the Kinova Gen3 arm
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to Kinova Gen3 arm at {self.ip_address}...")
            
            # Create TCP transport connection
            self.transport = TCPTransport()
            self.transport.connect(self.ip_address, 10000)  # Default port is 10000
            
            # Create router client with error callback
            def error_callback(exception):
                print(f"Router error: {exception}")
            
            self.router = RouterClient(self.transport, error_callback)
            
            # Create session
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.username
            session_info.password = self.password
            session_info.session_inactivity_timeout = 60000   # 60 seconds
            session_info.connection_inactivity_timeout = 2000 # 2 seconds
            
            # Session client
            self.session_client = SessionClient(self.router)
            self.session_client.CreateSession(session_info)
            
            # Create base client
            self.base_client = BaseClient(self.router)
            
            self.is_connected = True
            print(f"✓ Successfully connected to Kinova Gen3 arm at {self.ip_address}")
            
            # Get basic arm information
            self._print_arm_info()
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to Kinova Gen3 arm: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """
        Disconnect from the Kinova Gen3 arm and clean up resources
        """
        try:
            if self.session_client:
                self.session_client.CloseSession()
                print("Session closed")
            
            if self.router:
                self.router.SetActivationStatus(False)
                print("Router deactivated")
            
            if self.transport:
                self.transport.disconnect()
                print("Transport disconnected")
                
            self.is_connected = False
            print("✓ Disconnected from Kinova Gen3 arm")
            
        except Exception as e:
            print(f"Error during disconnection: {e}")

    def _print_arm_info(self):
        """
        Print basic information about the connected arm
        """
        try:
            if not self.is_connected or not self.base_client:
                return
                
            # Get product configuration
            product_config = self.base_client.GetProductConfiguration()
            print(f"Product: {product_config.product_code}")
            print(f"Model: {product_config.model}")
            print(f"Country: {product_config.country_code}")
            
            # Get arm state
            arm_state = self.base_client.GetArmState()
            print(f"Arm state: {Base_pb2.ArmState.Name(arm_state.active_state)}")
            
        except Exception as e:
            print(f"Could not retrieve arm information: {e}")

    def get_current_pose(self):
        """
        Get the current pose of the arm end-effector
        
        Returns:
            dict: Current pose with position (x,y,z) and orientation (theta_x, theta_y, theta_z)
        """
        if not self.is_connected:
            print("Not connected to arm")
            return None
            
        try:
            pose = self.base_client.GetMeasuredCartesianPose()
            return {
                'x': pose.x,
                'y': pose.y, 
                'z': pose.z,
                'theta_x': pose.theta_x,
                'theta_y': pose.theta_y,
                'theta_z': pose.theta_z
            }
        except Exception as e:
            print(f"Error getting current pose: {e}")
            return None

    def send_cartesian_delta(self, dpos, drot, duration=0.5):
        """
        Send a cartesian delta movement command
        
        Args:
            dpos (tuple): Delta position (dx, dy, dz) in meters
            drot (tuple): Delta rotation (dtheta_x, dtheta_y, dtheta_z) in degrees
            duration (float): Duration of movement in seconds
        """
        if not self.is_connected:
            print("ERROR: Not connected to arm")
            return False
        
        if not self.base_client:
            print("ERROR: Base client not initialized")
            return False
            
        try:
            print(f"DEBUG: Sending cartesian delta command...")
            print(f"  Position delta: {dpos} meters")
            print(f"  Rotation delta: {drot} degrees") 
            print(f"  Duration: {duration} seconds")
            
            # Create twist command (without duration - it's not part of TwistCommand)
            twist_command = Base_pb2.TwistCommand()
            twist_command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
            
            # Set linear velocities (converted from deltas)
            twist = twist_command.twist
            twist.linear_x = dpos[0] / duration
            twist.linear_y = dpos[1] / duration
            twist.linear_z = dpos[2] / duration
            
            # Set angular velocities (converted from deltas and degrees to rad/s)
            twist.angular_x = (drot[0] * 3.14159 / 180.0) / duration
            twist.angular_y = (drot[1] * 3.14159 / 180.0) / duration
            twist.angular_z = (drot[2] * 3.14159 / 180.0) / duration
            
            print(f"DEBUG: Linear velocities: x={twist.linear_x:.4f}, y={twist.linear_y:.4f}, z={twist.linear_z:.4f} m/s")
            print(f"DEBUG: Angular velocities: x={twist.angular_x:.4f}, y={twist.angular_y:.4f}, z={twist.angular_z:.4f} rad/s")
            
            # Check if the arm is in a safe state before sending command
            try:
                arm_state = self.base_client.GetArmState()
                print(f"DEBUG: Current arm state: {Base_pb2.ArmState.Name(arm_state.active_state)}")
                
                if arm_state.active_state == Base_pb2.ARMSTATE_IN_FAULT:
                    print("ERROR: Arm is in fault state. Cannot send movement command.")
                    return False
                    
            except Exception as state_error:
                print(f"WARNING: Could not check arm state: {state_error}")
            
            # Send command and wait for duration
            print("DEBUG: Sending twist command to arm...")
            self.base_client.SendTwistCommand(twist_command)
            print(f"DEBUG: Command sent successfully, waiting {duration} seconds...")
            
            # Wait for the specified duration, then stop the motion
            time.sleep(duration)
            self.base_client.Stop()
            print("DEBUG: Motion stopped after duration")
            
            print(f"✓ Sent delta movement: pos={dpos}, rot={drot}, duration={duration}s")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to send cartesian delta: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False

    def test_small_movements(self):
        """
        Test a series of small movements to validate the send_cartesian_delta function
        """
        if not self.is_connected:
            print("ERROR: Not connected to arm")
            return False
        
        print("\n" + "="*50)
        print("TESTING SMALL MOVEMENTS")
        print("="*50)
        
        # Get initial pose
        initial_pose = self.get_current_pose()
        if not initial_pose:
            print("ERROR: Could not get initial pose")
            return False
            
        print(f"Initial pose: x={initial_pose['x']:.3f}, y={initial_pose['y']:.3f}, z={initial_pose['z']:.3f}")
        
        # Test 1: Small movement in X direction
        print("\nTest 1: Moving +1cm in X direction")
        if self.send_cartesian_delta((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), duration=2.0):
            time.sleep(3)  # Wait for movement to complete
            new_pose = self.get_current_pose()
            if new_pose:
                dx = new_pose['x'] - initial_pose['x']
                print(f"  Actual movement: dx={dx:.4f}m (expected: 0.01m)")
        
        # Test 2: Return to original position
        print("\nTest 2: Returning to original position")
        if self.send_cartesian_delta((-0.01, 0.0, 0.0), (0.0, 0.0, 0.0), duration=2.0):
            time.sleep(3)
            final_pose = self.get_current_pose()
            if final_pose:
                dx = final_pose['x'] - initial_pose['x']
                print(f"  Final position error: dx={dx:.4f}m (expected: 0.0m)")
                
        print("="*50)
        return True

    def stop_motion(self):
        """
        Stop all motion of the arm
        """
        if not self.is_connected:
            print("Not connected to arm")
            return False
            
        try:
            self.base_client.Stop()
            print("Motion stopped")
            return True
        except Exception as e:
            print(f"Error stopping motion: {e}")
            return False

def main():
    """
    Main function to test the Kinova Gen3 connection
    """
    print("Kinova Gen3 Arm Connection Test")
    print("=" * 40)
    
    # Create controller instance
    # Update IP address as needed for your setup
    kinova = KinovaGen3Controller(ip_address="192.168.1.10")
    
    try:
        # Connect to arm
        if not kinova.connect():
            print("Failed to connect. Exiting.")
            return
        
        # Get current pose
        pose = kinova.get_current_pose()
        if pose:
            print("\nCurrent arm pose:")
            print(f"Position: x={pose['x']:.3f}, y={pose['y']:.3f}, z={pose['z']:.3f}")
            print(f"Orientation: θx={pose['theta_x']:.1f}°, θy={pose['theta_y']:.1f}°, θz={pose['theta_z']:.1f}°")
        
        # Example small movement (uncomment to test)
        print("\nSending small test movement...")
        kinova.send_cartesian_delta((0.00, -0.05, 0.0), (0.0, 0.0, 0.0), duration=2.0)
        
        print("\nConnection test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Always disconnect
        kinova.disconnect()

if __name__ == "__main__":
    main()