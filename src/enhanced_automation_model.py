from ultralytics import YOLO
import cv2
import math
import time
from datetime import datetime
import os 
# import pymssql
# import credential
# import numpy as np
# #----------------------------DATABASE CONNECTION______________________________________#
# conn = pymssql.connect(server=credential.server, user=credential.user, password=credential.password, database=credential.database)
# cursor = conn.cursor()
# g_temp_data = []


# DBVal = []
# for i in range(1, 16):
#     if(i%2 != 0):
#         temp = []
#         for j in range(1, 13):
#             temp.append(1)
#         DBVal.append(temp)
#     else:
#         temp = []
#         for j in range(1, 14):
#             temp.append(1)
#         DBVal.append(temp)

# DBTemp = []

# for i in range(1, 16):
#     if(i%2 != 0):
#         temp = []
#         for j in range(1, 13):
#             temp.append(1)
#         DBTemp.append(temp)
#     else:
#         temp = []
#         for j in range(1, 14):
#             temp.append(1)
#         DBTemp.append(temp)


def update_db(row, col, state):


    if(DBVal[row-1][col-1] != state):
        DBVal[row-1][col-1] = state
        cursor.execute ("UPDATE TBL_TUBE SET State = {0} WHERE phaseId = 2 AND RowNum = {1} and TubeNum = {2}".format(state, row, col))
        conn.commit()
    else:
       pass

    #cursor.execute ("UPDATE TBL_TUBE SET State = {0} WHERE phaseId = 2 AND RowNum = {1} and TubeNum = {2}".format(state, row, col))
    #conn.commit() 

def initDB():
    for i in range(1, 16):
        if(i % 2 != 0):
            for j in range(1, 13):
                cursor.execute ("UPDATE TBL_TUBE SET State = {0} WHERE phaseId = 2 AND phaseId = 2 AND RowNum = {1} and TubeNum = {2}".format(1, i, j))
                conn.commit()
        else:
            for j in range(1, 14):
                cursor.execute ("UPDATE TBL_TUBE SET State = {0} WHERE phaseId = 2 AND RowNum = {1} and TubeNum = {2}".format(1, i, j))
                conn.commit()

# Load the pretrained YOLO model
model = YOLO("best.pt")

# Define the color mapping for the labels
color_mapping = {
    0: "Yellow",  # 0 for yellow
    1: "Green",   # 1 for green
    2: "Blue",    # 2 for blue
    3: "Red",     # 3 for red
}
color_mapping_rect = {
    "Blue": (255, 0, 0),    # 0 for blue (BGR format)
    "Red": (0, 0, 255),    # 1 for red (BGR format)
    "Green": (0, 255, 0),    # 2 for green (BGR format)
    "Yellow": (0, 255, 255),  # 3 for yellow (BGR format)
}
color_mapping_class = {
    "Blue": 0,    # 0 for blue (BGR format)
    "Red": 1,    # 1 for red (BGR format)
    "Green": 2,    # 2 for green (BGR format)
    "Yellow": 3,  # 3 for yellow (BGR format)
}
color_mapping_db = {
    0: 4,
    1: 5,
    2: 3,
    3: 2
}

#find the possition of circle    
def find_position(number):
    # Define the number of elements in odd and even rows
    odd_row_length = 12
    even_row_length = 13
    
    # Loop through 15 rows
    current_number = 1  # The first number in the table starts at 1
    for row in range(1, 16):
        if row % 2 != 0:  # Odd rows
            row_length = odd_row_length
        else:  # Even rows
            row_length = even_row_length
        
        row_end = current_number + row_length - 1  # Last number in the current row
        
        if current_number <= number <= row_end:
            column = number - current_number + 1  # Calculate column within this row
            return row, column
        
        current_number = row_end + 1  # Update the number for the start of the next row

    return None  # In case the number is not found
# Function to check if a point (x, y) is inside a circle (x1, y1, radius)
def is_point_inside_circle(x, y, x1, y1, radius):
    distance = math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    return distance <= radius

# Function to read circle/tube coordinates from the .txt file
def read_circle_coordinates(file_path):
    circle_coordinates = []
    with open(file_path, "r") as file:
        for line in file:
            x1, y1, radius = map(int, line.strip().split(","))
            circle_coordinates.append((x1, y1, radius))
    return circle_coordinates

# Function to perform object detection and process the frame
def process_frame(frame, circle_coordinates, last_run_time, detected_data,data_dict, motion_level):
    current_time = time.time()
    if current_time - last_run_time >= 5 and motion_level < 50000:
        # Perform object detection on the frame
        results = model(frame)

        # Reset the detected data list
        detected_data.clear()
        for labels, (cx, cy) in data_dict.items():
            positions = find_position(labels)
            if positions:
                row, column = positions
                # update_db(row, column, 1)
        # Iterate over the detections
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            classes = result.boxes.cls.cpu().numpy()  # Get class IDs
            confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
            position_check = []
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = map(int, box)  # Convert box coordinates to integers
                label = int(cls)  # Get the class label
                center_x = (x1 + x2) // 2  # Calculate the center of the detected cap
                center_y = (y1 + y2) // 2
                for circle in circle_coordinates:
                    circle_x, circle_y, radius = circle
                    if is_point_inside_circle(center_x, center_y, circle_x, circle_y, radius):
                        detected_data.append((circle_x, circle_y, radius, color_mapping.get(label, "Unknown")))
                        break  # Stop checking other circles once a match is found
                
                for labels, (cx, cy) in data_dict.items():
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        
                        position = find_position(labels)
                        # print(len(position))
                        
                        if position:
                            position_check.append(position)
                            row, column = position
                            print(f"Detected At Number {labels} is in row {row}, column {column}.,,,,,{label}") # DNY
                            # update_db(row, column, color_mapping_db(label))
                        else:
                            print(f"Number {labels} is not in the table.")
                    elif x1 >= cx >= x2 and y1 >= cy >= y2:
                        position = find_position(labels)
                        row, column = position
                        # update_db(row, column, 1)
                print(len(position_check))
        # Update the last run time
        last_run_time = current_time

        # Draw bounding boxes and labels for detected caps (from the last YOLO run)
        for data in detected_data:
            circle_x, circle_y, radius, color = data
            # Draw the detected cap's bounding box and label
            cv2.rectangle(frame, (circle_x - radius, circle_y - radius), (circle_x + radius, circle_y + radius), color_mapping_rect[color], 2)
            cv2.putText(frame, str(color_mapping_class[color]), (circle_x - radius, circle_y - radius - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_mapping_rect[color], 2)

        # Save the frame with detections
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
        filename = os.path.join(output_folder, f"frame_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
        print(len(position_check))

        # # Print or save the detected data
        # for data in detected_data:
        #     circle_x, circle_y, radius, color = data
            # print(f"Circle: ({circle_x}, {circle_y}, {radius}), Detected Color: {str(color_mapping_class[color])}")
            # Optionally, save the data to a file 
            # with open(os.path.join(output_folder, f"detected_colors_{timestamp}.txt"), "a") as output_file:
            #     output_file.write(f"Circle: ({circle_x}, {circle_y}, {radius}), Detected Color: {str(color_mapping_class[color])}\n")

    return last_run_time, detected_data

# Main function
def main():
    # Read circle/tube coordinates from the .txt file
    circle_coordinates = read_circle_coordinates("tube_locations.txt")
    # initDB()
    # Open the video file
    video_path = "All_color_caps_inside_the_tube_under_bright_Lights.mp4"
    cap = cv2.VideoCapture(video_path)

    fgbg = cv2.createBackgroundSubtractorMOG2()
    # Create a named window for displaying the output
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)

    # Initialize variables for the 15-second timer
    last_run_time = time.time()  # Record the last time YOLO was run
    detected_data = []  # Store detected data from the last YOLO run
    # Initialize dictionary
    data_dict = {}

    # Open and read the file
    with open("tube_locations.txt", "r") as file:
        for idx, line in enumerate(file, start=1):  # Start keys from 1
            values = tuple(map(int, line.strip().split(",")))  # Convert to tuple of integers
            data_dict[idx] = values[:2]  # Store only the first two values

    # Print the dictionary
    print(data_dict)
    # Create a folder to save the frames
    global output_folder
    output_folder = "output_frames_new"
    os.makedirs(output_folder, exist_ok=True)

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        motion_level = cv2.countNonZero(fgmask)
        # Draw circles from the .txt file on the frame (white color)
        for circle in circle_coordinates:
            x1, y1, radius = circle
            cv2.circle(frame, (x1, y1), radius, (255, 255, 255), 2)  # White color, thickness 2

        # Process the frame
       
        last_run_time, detected_data = process_frame(frame, circle_coordinates, last_run_time, detected_data,data_dict,motion_level)

        # Display the frame in the output window
        cv2.imshow("Output", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()