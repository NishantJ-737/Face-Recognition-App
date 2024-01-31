import cv2
import numpy as np
import face_recognition
from datetime import datetime
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import os

class FaceRecognitionApp(App):
    def build(self):
        self.path = 'images'
        self.images = []
        self.classNames = []
        self.myList = os.listdir(self.path)
        for cl in self.myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])

        self.encodeListKnown = self.findEncodings(self.images)
        self.cap = None
        self.is_camera_running = False

        # GUI setup
        self.layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.result_label = Label(text='Recognition Result: Unknown', font_size=16, halign='center')
        self.layout.add_widget(self.result_label)

        self.image_widget = Image(size=(800, 600))
        self.layout.add_widget(self.image_widget)

        self.start_button = Button(text='Start Recognition', on_press=self.toggle_camera, font_size=16)
        self.layout.add_widget(self.start_button)

        self.history_label = Label(text='Recognition History:', font_size=16, halign='left')
        self.layout.add_widget(self.history_label)

        self.history_text = Label(text='', font_size=14, halign='left')
        self.layout.add_widget(self.history_text)

        self.recognition_history = []

        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return self.layout

    def findEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def markAttendance(self, name):
        with open('Attendance.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [entry.split(',')[0] for entry in myDataList]

            time_now = datetime.now()
            tString = time_now.strftime('%H:%M:%S')
            dString = time_now.strftime('%d/%m/%Y')

            entry_time_start = datetime.strptime("16:00:00", "%H:%M:%S").time()
            entry_time_end = datetime.strptime("17:59:00", "%H:%M:%S").time()
            exit_time_start = datetime.strptime("18:00:00", "%H:%M:%S").time()
            exit_time_end = datetime.strptime("19:30:00", "%H:%M:%S").time()

            f.seek(0, 2)  # Move the cursor to the end of the file

            if name not in nameList:  # Entry
                if entry_time_start <= time_now.time() <= entry_time_end:
                    # Record entry time in the 'Entry' column
                    entry = f'\n{name},Entry,{tString},{dString},'
                    f.write(entry)
                    f.seek(0)  # Move the cursor to the beginning of the file
                    myDataList = f.readlines()  # Read the file again
                    self.updateRecognitionHistory(entry)
                else:
                    # Record exit time in the 'Exit' column
                    entry = f'\n{name},Exit,{tString},{dString},'
                    f.write(entry)
                    f.seek(0)  # Move the cursor to the beginning of the file
                    myDataList = f.readlines()  # Read the file again
                    self.updateRecognitionHistory(entry)
            else:  # Exit (assuming only one entry per day)
                for i, line in enumerate(myDataList):
                    entry = line.split(',')
                    if entry[0] == name and entry[-1].strip() == "":
                        if exit_time_start <= time_now.time() <= exit_time_end:
                            # Check if "Exit" is already in the line
                            if "Exit" not in line:
                                # Append exit time in the same row without overwriting entry details
                                entry = f'{line.strip()},Exit,{tString},{dString}\n'
                                myDataList[i] = entry
                                f.seek(0)
                                f.writelines(myDataList)  # Write modified data back to the file
                                self.updateRecognitionHistory(entry)
                        break

    def updateRecognitionHistory(self, entry):
        self.recognition_history.append(entry.strip())

        if len(self.recognition_history) > 5:
            self.recognition_history.pop(0)

        self.history_text.text = '\n'.join(self.recognition_history)



    def update(self, dt):
        if self.is_camera_running:
            success, img = self.cap.read()
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS)
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    self.result_label.text = f'Recognition Result: {name}'
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    self.markAttendance(name)
                else:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 250, 0), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    self.result_label.text = 'Recognition Result: Unknown'

            buf1 = cv2.flip(img, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(img.shape[1], img.shape[0]), colorfmt='bgr')
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture1

    def toggle_camera(self, instance):
        if self.is_camera_running:
            self.cap.release()
            self.is_camera_running = False
            self.start_button.text = 'Start Recognition'
            self.result_label.text = 'Recognition Result: Unknown'
        else:
            self.cap = cv2.VideoCapture(0)
            self.is_camera_running = True
            self.start_button.text = 'Stop Recognition'

if __name__ == '__main__':
    FaceRecognitionApp().run()
