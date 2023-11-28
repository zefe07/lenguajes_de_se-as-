import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 27
dataset_size = 100

# Initialize the camera
cap = cv2.VideoCapture(-1)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class {}'.format(j))

    done = False
    while not done:
        ret, frame = cap.read()

        # Check if the frame is valid
        if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.putText(frame, 'Listo? Presione "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                done = True
                break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()

        # Check if the frame is valid
        if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(
                j), '{}.jpg'.format(counter)), frame)

            counter += 1

# Release the camera capture object
cap.release()
cv2.destroyAllWindows()
