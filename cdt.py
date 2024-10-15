import cv2
import numpy as np

# Function to detect and track a specific color
def detect_and_track_color():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return
    
    while True:
        # Read the current frame from the webcam
        ret, frame = cap.read()

        # If the frame was not successfully captured, break out of the loop
        if not ret:
            break

        # Convert the frame from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the color range for detection (in HSV color space)
        # Example: Detecting a blue color (you can modify this for different colors)
        lower_bound = np.array([100, 150, 50])  # Lower HSV bound for blue
        upper_bound = np.array([140, 255, 255])  # Upper HSV bound for blue

        # Create a mask that extracts the color within the bounds
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Use morphological operations to remove small noises from the mask
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours
        for contour in contours:
            # If the contour is large enough, draw a rectangle around it
            if cv2.contourArea(contour) > 500:
                # Get the bounding box coordinates
                x, y, w, h = cv2.boundingRect(contour)
                # Draw a rectangle around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Optionally, draw the center of the object
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Show the original frame with tracking
        cv2.imshow("Color Detection and Tracking", frame)

        # Break the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_and_track_color()
