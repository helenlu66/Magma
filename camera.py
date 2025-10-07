# --- Camera ---
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
import sys
import os
import imageio
from datetime import datetime

# Set DISPLAY for showing images
os.environ['DISPLAY'] = ':0'

# Check if we have display capability
HAS_DISPLAY = 'DISPLAY' in os.environ

print(f"DISPLAY environment variable: {os.environ.get('DISPLAY', 'Not set')}")
print(f"HAS_DISPLAY: {HAS_DISPLAY}")

# Test if we can actually create windows
try:
    import cv2
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imshow("Test Window", test_img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("Display test: SUCCESS - Can create OpenCV windows")
    CAN_DISPLAY = True
except Exception as e:
    print(f"Display test: FAILED - {e}")
    CAN_DISPLAY = False

# Video writer setup for when display is not available
video_writer = None
VIDEO_FILENAME = f"camera_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
VIDEO_FPS = 30
VIDEO_FRAME_SIZE = (640, 480)

# --- Model (pseudocode placeholders) ---
# magma_model, magma_processor = load_magma(...)
# def magma_infer(pil_image): return action_tokens

# --- Kinova (Kortex API) ---
# from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
# from kortex_api.autogen.messages import Base_pb2
# from kortex_api.autogen.client_stubs.SessionClientRpc import SessionClient
# from kortex_api.autogen.messages import Session_pb2
# from kortex_api import RouterClient, DeviceConfig, DeviceConnection
# import argparse

# def kinova_connect(ip="192.168.1.10", username="admin", password="admin"):
#     """
#     Connect to Kinova robot and return BaseClient
#     """
#     try:
#         # Create connection to the device
#         device_connection = DeviceConnection.DeviceConnection(ip_address=ip)
        
#         # Create router client
#         router = RouterClient.RouterClient(device_connection, lambda kException: print("_________ callback error _________ = {}".format(kException)))
        
#         # Create session
#         session_info = Session_pb2.CreateSessionInfo()
#         session_info.username = username
#         session_info.password = password
#         session_info.session_inactivity_timeout = 60000   # (milliseconds)
#         session_info.connection_inactivity_timeout = 2000 # (milliseconds)
        
#         # Session client
#         session_client = SessionClient(router)
#         session_client.CreateSession(session_info)
        
#         # Create base client
#         base_client = BaseClient(router)
        
#         print(f"Successfully connected to Kinova robot at {ip}")
#         return base_client, router, session_client
        
#     except Exception as e:
#         print(f"Failed to connect to Kinova robot: {e}")
#         return None, None, None

# def send_cartesian_delta(base: BaseClient, dpos, drot, gripper_open=None, duration=0.2):
#     """
#     Send cartesian delta movement command to Kinova robot
#     dpos = (dx,dy,dz) in meters in the chosen base frame
#     drot = (droll, dpitch, dyaw) in radians
#     """
#     try:
#         cmd = Base_pb2.TwistCommand()
#         cmd.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL
#         cmd.duration = duration  # seconds
#         twist = cmd.twist
#         twist.linear_x, twist.linear_y, twist.linear_z = dpos
#         twist.angular_x, twist.angular_y, twist.angular_z = drot
#         base.SendTwistCommand(cmd)
#         # gripper control via GripperCommand if needed
#         if gripper_open is not None:
#             # Add gripper control logic here if needed
#             pass
#     except Exception as e:
#         print(f"Error sending cartesian delta: {e}")

def save_camera_video(frames, path, video_name="camera_feed"):
    """Saves a video replay of camera frames."""
    if path != ".":
        os.makedirs(path, exist_ok=True)
    processed_video_name = video_name.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    video_path = f"{path}/{processed_video_name}.mp4"
    video_writer = imageio.get_writer(video_path, fps=30)
    for img in frames:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved camera video at path {video_path}")
    return video_path

# --- Main loop ---
try:
    # Try to initialize RealSense camera
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    print("RealSense camera initialized successfully")
except Exception as e:
    print(f"Failed to initialize RealSense camera: {e}")
    print("Camera is busy or not available. Exiting.")
    sys.exit(1)

# Connect to Kinova robot
# base, router, session = kinova_connect()
# if base is None:
#     print("Warning: Could not connect to Kinova robot. Continuing with camera only.")
#     KINOVA_CONNECTED = False
# else:
#     print("Kinova robot connected successfully!")
#     KINOVA_CONNECTED = True

frame_count = 0
captured_frames = []  # Store frames for video creation

try:
    print(f"Starting camera loop. Display available: {HAS_DISPLAY and CAN_DISPLAY}")
    if not (HAS_DISPLAY and CAN_DISPLAY):
        print(f"Running in video recording mode - will save to {VIDEO_FILENAME}")
        print("Press Ctrl+C to stop recording")
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, VIDEO_FPS, VIDEO_FRAME_SIZE)
        if not video_writer.isOpened():
            print("Error: Could not open video writer")
            sys.exit(1)
        print(f"Video writer initialized: {VIDEO_FILENAME}")
        
    while True:  # Run continuously until Ctrl+C
        # Use real RealSense camera
        frames = pipe.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            continue
        rgb = np.asanyarray(color.get_data())

        pil_img = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        frame_count += 1
        
        # Store frame for video creation (convert BGR to RGB for imageio)
        captured_frames.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        # 1) MAGMA inference → action tokens
        # tokens = magma_infer(pil_img)

        # 2) Decode tokens → metric deltas (example placeholder)
        # dx,dy,dz, droll,dpitch,dyaw, grip = decode_magma(tokens)

        # 3) Safety clamp & rate limit
        # dx,dy,dz = clamp_vec((dx,dy,dz), max_step=0.01)  # 1 cm/step
        # droll,dpitch,dyaw = clamp_vec((droll,dpitch,dyaw), max_step=0.05)

        # 4) Send to Kinova (if connected)
        # if KINOVA_CONNECTED and base is not None:
        #     # Example: send small movements (uncomment when you have actual action predictions)
        #     # send_cartesian_delta(base, (dx,dy,dz), (droll,dpitch,dyaw), gripper_open=(grip>0.5))
        #     pass

        # 5) (Optional) visualize
        if HAS_DISPLAY and CAN_DISPLAY:
            # Add frame number to image to make it more obvious
            display_img = rgb.copy()
            cv2.putText(display_img, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "Press ESC to exit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("RealSense RGB Camera", display_img)
            cv2.moveWindow("RealSense RGB Camera", 100, 100)  # Position window
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        else:
            # Save to video when display is not available
            if video_writer is not None:
                # Add frame number overlay to video
                video_img = rgb.copy()
                cv2.putText(video_img, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(video_img, f"Recording: {VIDEO_FILENAME}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                video_writer.write(video_img)
                
                # Print progress every 30 frames
                if frame_count % 30 == 0:
                    print(f"Recording frame {frame_count} to video...")
            
            # Add a small delay to maintain reasonable frame rate
            import time
            time.sleep(1.0 / VIDEO_FPS)
            
except KeyboardInterrupt:
    print(f"\nInterrupted by user after {frame_count} frames")
finally:
    # print(f"Processed {frame_count} frames total")
    # pipe.stop()
    
    # # Clean up Kinova connection
    # if 'session' in locals() and session is not None:
    #     try:
    #         session.CloseSession()
    #         print("Kinova session closed")
    #     except:
    #         pass
    # if 'router' in locals() and router is not None:
    #     try:
    #         router.SetActivationStatus(False)
    #         print("Kinova router deactivated")
    #     except:
    #         pass
    
    # Save captured frames as video using the save_camera_video function
    if captured_frames:
        video_save_path = "."
        video_path = save_camera_video(captured_frames, video_save_path, f"camera_feed_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print(f"Video saved using save_camera_video function: {video_path}")
    
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved as: {VIDEO_FILENAME}")
    if HAS_DISPLAY and CAN_DISPLAY:
        cv2.destroyAllWindows()
    print("Camera stopped and cleanup completed")
