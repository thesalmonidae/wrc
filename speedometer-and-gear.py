import cv2
import pytesseract
import numpy as np

# Set up Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the OBS Virtual Camera or capture device
cap = cv2.VideoCapture(2)  # Use the correct index for your OBS Virtual Camera

# Set the resolution to match OBS output if necessary
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Define ROIs (adjust based on your game HUD layout)
    speedometer_roi = frame[832:880, 1575:1675]
    gear_roi = frame[735:820, 1575:1675]  # Adjust coordinates for gear indicator

    # Process the speedometer ROI
    speed_gray = cv2.cvtColor(speedometer_roi, cv2.COLOR_BGR2GRAY)
    speed_thresh = cv2.adaptiveThreshold(speed_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Use OCR to read the speed
    speed_text = pytesseract.image_to_string(speed_thresh, config='--psm 6 -c tessedit_char_whitelist=0123456789')
    try:
        speed = int(speed_text.strip())
        print("Detected Speed:", speed)
    except ValueError:
        print("Could not read speed")

    # Process the gear ROI
    gear_gray = cv2.cvtColor(gear_roi, cv2.COLOR_BGR2GRAY)
    #gear_thresh = cv2.adaptiveThreshold(gear_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, gear_thresh = cv2.threshold(gear_gray, 100, 255, cv2.THRESH_BINARY)  # Adjust 100 based on lighting


    # Use OCR to read the gear
    gear_text = pytesseract.image_to_string(gear_thresh, config='--psm 6 -c tessedit_char_whitelist=12345NR')
    try:
        gear = int(gear_text.strip())
        print("Detected Gear:", gear)
    except ValueError:
        print("Could not read gear")

    # Display the ROIs for debugging (optional)
    cv2.imshow('Speedometer ROI', speedometer_roi)
    cv2.imshow('Processed Speedometer', speed_thresh)
    cv2.imshow('Gear ROI', gear_roi)
    cv2.imshow('Processed Gear', gear_thresh)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
