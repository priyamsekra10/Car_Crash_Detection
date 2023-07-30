import cv2
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
frame_interval = 0.1
import time
last_emergency_response_time = 0

import pymongo
from pyfcm import FCMNotification
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
#---------------------------------------------------------------------------------------------------------------------
# will come from dashcam
video_path = r"C:\Users\priya\Desktop\crash_working.mp4"
email_id = 'priyam22rr@gmail.com'
# time = t
#---------------------------------------------------------------------------------------------------------------------
IMG_SIZE = 224

def emergency_response(email_id):
    client = pymongo.MongoClient("mongodb+srv://priyam:pqrs.123@cluster0.1uefwpt.mongodb.net/")
    db = client["car_crash"]
    collection = db["user_login"]
    
    def import_data_by_email(email_id):
        try:
            data_cursor = collection.find({"email_id": email_id})
            df = pd.DataFrame(list(data_cursor))

            if not df.empty:
                return df
            else:
                print("No data found for the provided email ID.")
                return None

        except Exception as e:
            print("Error occurred while importing data:", e)
            return None
        finally:
            client.close()

    email_id_to_search = email_id
    result_df = import_data_by_email(email_id_to_search)

    if result_df is not None:
        print("Data imported successfully for email:", email_id_to_search)
        
    for i in range(len(result_df['_id'])):
        fcm = result_df['fcm'][i]
        fcm
        
    def notify_crash(fcm_token, crash_info):
        push_service = FCMNotification(api_key="AAAAucIfw-w:APA91bHy03w5pMy4AVf14qKy7M1Bw0JXMm4_A19r_KuY1viHVL3ky7wsqa34oaceDCTsQWaB5dGwa4gnDDqDnch9VvRjcl-fQw1YAY_WxNvhtigD5NGDftJEUSKJMp2ePWd3pQGS_UNm")

        message_title = "Crash Detected"
        message_body = "A crash was detected at location X."

        result = push_service.notify_single_device(
            registration_id=fcm_token, 
            message_title=message_title, 
            message_body=message_body, 
            data_message=crash_info
        )

    notify_crash(fcm_token= fcm ,crash_info = {
        'crash_time': '2021-07-11 14:30:00',
        'crash_location': 'Lat: 40.7128, Lon: 74.0060',
        'crash_severity': 'High',
        # ... any other data you want to send ...
    })

    print("Done")
    
    
#     email:

    for i in range(len(result_df['r_email'])):
        receiver_email = result_df['r_email'][i]
        
    def send_email(sender_email, receiver_email):
    #     sender_email = 'priyam22rr@gmail.com'
    #     receiver_email = 'shivamvijayvargiya03@gmailcom'
        subject = 'crash detected'
        message = 'crash alert!'

        # Create a multipart message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject

        # Add the message body
        msg.attach(MIMEText(message, 'plain'))

        # SMTP server configuration
        smtp_server = 'smtp.gmail.com'
        smtp_port = 587  # Replace with the appropriate port for your server

        # Establish a secure connection with the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            # Login to your email account
            server.login(sender_email, 'mlonvvatlfnilogu')
            # Send the email
            server.send_message(msg)

    send_email(sender_email = email_id, receiver_email = receiver_email)
    print("email sent")
        
# emergency_response(email_id = email_id)

def can_call_emergency_response():
    global last_emergency_response_time
    current_time = time.time()
    return (current_time - last_emergency_response_time) >= 300  # 5 minutes in seconds

def call_emergency_response():
    global last_emergency_response_time
    emergency_response(email_id=email_id)
    last_emergency_response_time = time.time()
    print("function called")
    print("=============-=-=-=======================-=-=-===================-=-=-===")
    
def load_crash_detection_model(model_path):
    # Load the crash detection model with custom objects
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = tf.image.resize(frame_rgb, size=[IMG_SIZE, IMG_SIZE])
    frame_tensor = tf.convert_to_tensor(frame_resized, dtype=tf.float32) / 255.0
    frame_batch = tf.expand_dims(frame_tensor, axis=0)
    return frame_batch

def predict_crash(model, frame_batch):
    prediction = model.predict(frame_batch)
    return prediction

def extract_frames(video_path, frame_interval, model):
    video = cv2.VideoCapture(video_path)

    frame_count = 0
    success = True
    next_extraction_time = frame_interval

    frame_info_list = []

    while success:
        success, frame = video.read()

        if success:
            frame_rate = video.get(cv2.CAP_PROP_FPS)
            current_time = frame_count / frame_rate

            if current_time >= next_extraction_time:
                frame_batch = process_frame(frame)

                prediction = predict_crash(model, frame_batch)
                print(prediction)
                crash_max = prediction[0][1]
                
                if crash_max > 0.80:
                    if can_call_emergency_response():
                        call_emergency_response()
                else:
                    continue

                frame_info = {
                    "Frame": frame_count,
                    "Time (seconds)": current_time,
                    "Crash Prediction": prediction[0][1],
                }

                frame_info_list.append(frame_info)

                next_extraction_time += frame_interval

            frame_count += 1

    video.release()
    df = pd.DataFrame(frame_info_list)
    return df


crash_detection_model_path = 'car.h5'
crash_detection_model = load_crash_detection_model(crash_detection_model_path)

extract_frames(video_path, frame_interval, crash_detection_model)
