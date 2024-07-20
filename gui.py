import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2


def load_model(model_path, weights_path):
    with open(model_path, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



top = tk.Tk()
top.geometry('800x600')
top.title("Car and People Detector")
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


car_model = load_model(r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 3\model_a.json",
                       r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 3\Model_weights.weight.h5")


people_model = load_model(r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 3\model_a.json",
                          r"C:\Users\LENEVO\OneDrive\Desktop\Assignment 3\Model_weights.weight.h5")


def preprocess_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def swap_colors(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
   
    red_lower = np.array([0, 70, 50])
    red_upper = np.array([10, 255, 255])
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])

    s
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    
    image[red_mask > 0] = [255, 0, 0]
    image[blue_mask > 0] = [0, 0, 255]

    return image


def detect_and_swap(file_path):
    global label_packed

    image = cv2.imread(file_path)
    input_image = preprocess_image(image, (448, 448))

    # Detect cars
    car_predictions = car_model.predict(input_image)
    draw_bounding_boxes(image, car_predictions, color_swap=True)

    # Detect people
    people_predictions = people_model.predict(input_image)
    count_people(image, people_predictions)

    # Display the processed image
    display_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    im = ImageTk.PhotoImage(display_image)

    sign_image.configure(image=im)
    sign_image.image = im
    label1.configure(text='Detection Complete')


def draw_bounding_boxes(image, predictions, color_swap=False):
    grid_size = 7  
    num_anchors = 3  
    box_confidence_threshold = 0.5  

    predictions = predictions.reshape((grid_size, grid_size, num_anchors, -1))
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(num_anchors):
                box_confidence = predictions[i, j, k, 0]
                if box_confidence > box_confidence_threshold:
                    box_x, box_y, box_w, box_h = predictions[i, j, k, 1:5]

                    box_x = int((i + box_x) * (image.shape[1] / grid_size))
                    box_y = int((j + box_y) * (image.shape[0] / grid_size))
                    box_w = int(box_w * image.shape[1])
                    box_h = int(box_h * image.shape[0])
                    x1 = max(0, box_x - box_w // 2)
                    y1 = max(0, box_y - box_h // 2)
                    x2 = min(image.shape[1], box_x + box_w // 2)
                    y2 = min(image.shape[0], box_y + box_h // 2)

                    if color_swap:
                        car_image = image[y1:y2, x1:x2]
                        swapped_car_image = swap_colors(car_image)
                        image[y1:y2, x1:x2] = swapped_car_image

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def count_people(image, predictions):
    grid_size = 7  
    num_anchors = 3  
    box_confidence_threshold = 0.5  

    predictions = predictions.reshape((grid_size, grid_size, num_anchors, -1))
    male_count = 0
    female_count = 0

    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(num_anchors):
                box_confidence = predictions[i, j, k, 0]
                if box_confidence > box_confidence_threshold:
                    gender_prediction = np.argmax(predictions[i, j, k, 5:])
                    if gender_prediction == 0:  
                        male_count += 1
                    else:
                        female_count += 1

    label1.configure(text=f'Males: {male_count}, Females: {female_count}')


def show_detect_button(file_path):
    detect_b = Button(top, text="Detect Objects", command=lambda: detect_and_swap(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(e)


upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Object Detector', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()
top.mainloop()
