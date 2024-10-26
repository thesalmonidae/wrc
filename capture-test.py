import cv2

# Open the OBS Virtual Camera (use the correct index, e.g., 2)
cap = cv2.VideoCapture(2)

# Set the desired width and height (match OBS output resolution)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Replace with your OBS width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Replace with your OBS height

if not cap.isOpened():
    print("Error: Could not open OBS Virtual Camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Display the frame
    cv2.imshow('OBS Virtual Camera Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
