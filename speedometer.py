import cv2
import pytesseract

# Set up Tesseract OCR path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

# Open the OBS Virtual Camera or capture device
cap = cv2.VideoCapture(2)  # Use the correct index for your OBS Virtual Camera

# Set the desired resolution (match your OBS output)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # OBS width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # OBS height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Define the ROI for the speedometer (adjust coordinates for your game HUD)
    speedometer_roi = frame[835:880, 1575:1675]  # Adjust based on speedometer position

    # Preprocess the ROI for better OCR results
    gray = cv2.cvtColor(speedometer_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # Apply thresholding

    # Use Tesseract OCR to extract text (only digits) from the preprocessed ROI
    speed_text = pytesseract.image_to_string(thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')

    # Try to convert the OCR result to an integer
    try:
        speed = int(speed_text)
        print("Speed:", speed)
    except ValueError:
        print("Could not read speed")

    # Display the ROI and the thresholded image for debugging
    cv2.imshow('Speedometer', speedometer_roi)
    cv2.imshow('Thresholded Speedometer', thresh)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
import cv2
import pytesseract
import numpy as np

# Set up Tesseract OCR path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust path as needed

# Open the OBS Virtual Camera or capture device
cap = cv2.VideoCapture(2)  # Use the correct index for your OBS Virtual Camera

# Set the desired resolution (match your OBS output)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # OBS width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # OBS height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Define the ROI for the speedometer (adjust coordinates for your game HUD)
    speedometer_roi = frame[820:890, 1550:1700]  # Adjust based on speedometer position

    # Preprocess the ROI for better OCR results
    gray = cv2.cvtColor(speedometer_roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Try adaptive thresholding for better results with variable lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Use OCR to read the speed (allowing for multiple digits)
    speed_text = pytesseract.image_to_string(sharpened, config='--psm 6 -c tessedit_char_whitelist=0123456789')

    # Clean up and parse the OCR result
    speed_text = speed_text.replace(" ", "").replace("\n", "")
    try:
        speed = int(speed_text)
        print("Speed:", speed)
    except ValueError:
        print("Could not read speed")

    # Display the ROI and the thresholded image for debugging
    cv2.imshow('Speedometer', speedometer_roi)
    cv2.imshow('Processed Speedometer', sharpened)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
