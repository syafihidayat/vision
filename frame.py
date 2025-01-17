import pyrealsense2 as rs
import cv2
import numpy as np

# Global variables
lowhuepadi, sminpadi, vminpadi = 164, 105, 0
upperhuepadi, smaxpadi, vmaxpadi = 178, 255, 255
minDistance, maxDistance = 500, 1500
threshold_value = 255

# Create background subtractor
pBackSub = cv2.createBackgroundSubtractorKNN()

def setup_trackbars():
    cv2.namedWindow("Trackbars", cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Low Hue", "Trackbars", lowhuepadi, 255, lambda x: None)
    cv2.createTrackbar("High Hue", "Trackbars", upperhuepadi, 255, lambda x: None)
    cv2.createTrackbar("Min Distance", "Trackbars", minDistance, 2000, lambda x: None)
    cv2.createTrackbar("Max Distance", "Trackbars", maxDistance, 2000, lambda x: None)
    cv2.createTrackbar("Threshold", "Trackbars", threshold_value, 255, lambda x: None)

def process_frame(frame, depth):
    global lowhuepadi, sminpadi, vminpadi, upperhuepadi, smaxpadi, vmaxpadi, minDistance, maxDistance

    # Update trackbar values
    lowhuepadi = cv2.getTrackbarPos("Low Hue", "Trackbars")
    upperhuepadi = cv2.getTrackbarPos("High Hue", "Trackbars")
    minDistance = cv2.getTrackbarPos("Min Distance", "Trackbars")
    maxDistance = cv2.getTrackbarPos("Max Distance", "Trackbars")

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask based on HSV range
    lower = np.array([lowhuepadi, sminpadi, vminpadi])
    upper = np.array([upperhuepadi, smaxpadi, vmaxpadi])
    mask = cv2.inRange(hsv, lower, upper)

    # Create depth masks
    object_mask = (depth > minDistance).astype(np.uint8)
    object_mask2 = (depth < maxDistance).astype(np.uint8)

    # Combine masks
    combined_mask = cv2.bitwise_and(mask, mask, mask=object_mask2)

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_mask = cv2.erode(combined_mask, kernel)
    combined_mask = cv2.dilate(combined_mask, kernel)

    # Draw centerlines
    y_position = 240  # Frame height / 2
    x_position = 320  # Frame width / 2
    cv2.line(frame, (0, y_position), (frame.shape[1], y_position), (0, 0, 0), 2)
    cv2.line(frame, (x_position, 0), (x_position, frame.shape[0]), (0, 0, 0), 2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 200 < area < 10000:
            bounding_rect = cv2.boundingRect(contour)
            cv2.rectangle(frame, bounding_rect, (0, 255, 0), 2)

            center = (bounding_rect[0] + bounding_rect[2] // 2, bounding_rect[1] + bounding_rect[3] // 2)
            cv2.circle(frame, center, 3, (255, 255, 255), -1)

            dist = depth[center[1], center[0]]
            print(f"Bounding Box Center: {center}")

    # Display results
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Frame", frame)

def padi_detection():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    pipeline.start(config)
    setup_trackbars()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Error: Frames not captured correctly!")
                continue

            # Convert frames to numpy arrays
            depth = np.asanyarray(depth_frame.get_data())
            frame = np.asanyarray(color_frame.get_data())

            process_frame(frame, depth)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    padi_detection()
