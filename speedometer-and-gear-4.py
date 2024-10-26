import cv2
import pytesseract
import numpy as np

# Set up Tesseract OCR path for speed detection
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load gear templates (1-5, N, R) and ensure they are grayscale
gear_filenames = {
    '1': 'gear_1.png',
    '2': 'gear_2.png',
    '3': 'gear_3.png',
    '4': 'gear_4.png',
    '5': 'gear_5.png',
    'N': 'gear_N.png',
    'R': 'gear_R.png'
}

gear_templates = {}
for gear, filename in gear_filenames.items():
    # Load each template
    template = cv2.imread(filename)
    
    # Ensure the template is converted to grayscale
    if template is not None:
        if len(template.shape) == 3:  # If image is not grayscale
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        gear_templates[gear] = template
    else:
        print(f"Error: Could not load template for gear '{gear}' from file {filename}")

# Open the OBS Virtual Camera or capture device
cap = cv2.VideoCapture(2)  # Use the correct index for your OBS Virtual Camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Adjust as needed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # ---- Speed Detection with OCR ----
    # Define the ROI for the speedometer (adjust coordinates based on your game HUD)
    speedometer_roi = frame[790:830, 1575:1680]

    # Convert speedometer ROI to grayscale and apply thresholding
    speed_gray = cv2.cvtColor(speedometer_roi, cv2.COLOR_BGR2GRAY)
    _, speed_thresh = cv2.threshold(speed_gray, 100, 255, cv2.THRESH_BINARY)

    # Use Tesseract OCR to detect speed
    speed_text = pytesseract.image_to_string(speed_thresh, config='--psm 8 -c tessedit_char_whitelist=0123456789')
    try:
        speed = int(speed_text.strip())
        print("Detected Speed:", speed)
    except ValueError:
        print("Could not read speed")

    # ---- Gear Detection with Template Matching ----
    # Define the ROI for the gear indicator (adjust coordinates based on gear position)
    gear_roi = frame[690:775, 1575:1675]

    # Convert gear ROI to grayscale and apply binary thresholding
    gear_gray = cv2.cvtColor(gear_roi, cv2.COLOR_BGR2GRAY)
    _, gear_thresh = cv2.threshold(gear_gray, 100, 255, cv2.THRESH_BINARY)

    # Perform template matching for each gear template
    best_match = None
    highest_score = float('-inf')
    detected_gear = None

    for gear, template in gear_templates.items():
        # Match template with the gear ROI
        res = cv2.matchTemplate(gear_thresh, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        # Track the best matching gear based on the highest score
        if max_val > highest_score:
            highest_score = max_val
            detected_gear = gear

    # Output the detected gear based on the best match
    if detected_gear is not None and highest_score > 0.8:  # Use confidence threshold (0.8) to filter results
        print("Detected Gear:", detected_gear)
    else:
        print("Could not detect gear")

    # ---- Display for Debugging ----
    # Display the ROIs and processed images for speed and gear detection
    cv2.imshow('Speedometer ROI', speedometer_roi)
    cv2.imshow('Processed Speedometer', speed_thresh)
    cv2.imshow('Gear ROI', gear_roi)
    cv2.imshow('Processed Gear', gear_thresh)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
