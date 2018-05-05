
import sys
import os

if sys.version_info[0] < 3 and not sys.platform.startswith("win"):
    from Tkinter import Tk, Label, Button, Entry, IntVar, END, W, E, StringVar
    import ttk
else:
    from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E, ttk, StringVar
if sys.version_info[0] < 3:
    from tkFileDialog import askopenfilename
else:
    from tkinter.filedialog import askopenfilename

import numpy as np
import cv2
import time
import dlib

class FaceDetector:

    def __init__(self, master):
        self.master = master
        master.title("FaceDetector")

        self.label = Label(master, text="Choose and run the face detector you want")
        self.label.grid(row=0, column=0, columnspan=3)
        
        self.filename = os.path.join(os.path.dirname(__file__), "data", "mcem0_head.mpg")
        
        def video_stream_var_to_filename(event):
            if self.video_stream_var.get() == "default video":
                self.filename = os.path.join(os.path.dirname(__file__), "data", "mcem0_head.mpg")
                self.video_stream_file_path_var.set("")
            if self.video_stream_var.get() == "webcam":
                self.filename = 0
                self.video_stream_file_path_var.set("")
            if self.video_stream_var.get() == "open new file":
                self.filename = askopenfilename(parent=root)
                self.video_stream_file_path_var.set("Path to current file: "+self.filename)
        
        self.video_stream_label = Label(master, text="Video stream")
        self.video_stream_label.grid(row=1, column=0)
        self.video_stream_var = StringVar()
        self.video_stream_var.set("default video")
        self.video_stream = ttk.Combobox(master, textvariable=self.video_stream_var, state="readonly")
        self.video_stream['values'] = ["default video", "webcam", "open new file"]
        self.video_stream.bind("<<ComboboxSelected>>", video_stream_var_to_filename)
        self.video_stream.grid(row=1, column=2, columnspan=1)
        
        self.video_stream_file_path_var = StringVar()
        self.video_stream_file_path_text = Label(master, textvariable=self.video_stream_file_path_var)
        self.video_stream_file_path_text.grid(row=1, column=3, columnspan=7, sticky=W)

        self.model_type_label = Label(master, text="Model type")
        self.model_type_label.grid(row=2, column=0)
        
        self.model_label = Label(master, text="Model")
        self.model_label.grid(row=3, column=0)
        
        self.parameters_label = Label(master, text="Parameters")
        self.parameters_label.grid(row=4, column=0)
        
        self.cascades_avg_time_per_frame_label = Label(master, text="Average time per frame")
        self.cascades_avg_time_per_frame_label.grid(row=8, column=0)
        
        self.cascades_avg_fps_label = Label(master, text="Average FPS number")
        self.cascades_avg_fps_label.grid(row=9, column=0)
        
        # cascades
        self.model_type_cascades_label = Label(master, text="Cascades")
        self.model_type_cascades_label.grid(row=2, column=2, columnspan=2)
        
        self.model_cascades_var = StringVar()
        self.model_cascades_var.set("haarcascade_frontalface_alt")
        self.model_cascades = ttk.Combobox(master, textvariable=self.model_cascades_var, state="readonly", width=30)
        self.model_cascades['values'] = ("haarcascade_frontalface_default",
                                "haarcascade_frontalface_alt",
                                "haarcascade_frontalface_alt2",
                                "haarcascade_frontalface_alt_tree",
                                "lbpcascade_frontalface",
                                "lbpcascade_frontalface_improved")
        self.model_cascades.grid(row=3, column=2, columnspan=2)

        self.cascades_scale_factor_label = Label(master, text="Scale factor")
        self.cascades_scale_factor_label.grid(row=4, column=2)
        
        self.cascades_scale_factor_var = StringVar()
        self.cascades_scale_factor_var.set(str(1.1))
        self.cascades_scale_factor = ttk.Combobox(master, textvariable=self.cascades_scale_factor_var, state="readonly")
        self.cascades_scale_factor['values'] = [str(x) for x in [1.01, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.5, 3]]
        self.cascades_scale_factor.grid(row=4, column=3, columnspan=1)

        self.cascades_min_neighbors_label = Label(master, text="Minimum number of neighbors")
        self.cascades_min_neighbors_label.grid(row=5, column=2)
        
        self.cascades_min_neighbors_var = IntVar()
        self.cascades_min_neighbors_var.set(3)
        self.cascades_min_neighbors = ttk.Combobox(master, textvariable=self.cascades_min_neighbors_var, state="readonly")
        self.cascades_min_neighbors['values'] = range(1,21)
        self.cascades_min_neighbors.grid(row=5, column=3, columnspan=1)

        self.cascades_min_size_x_label = Label(master, text="Minimum size of a face (x axis)")
        self.cascades_min_size_x_label.grid(row=6, column=2)
        
        self.cascades_min_size_x_var = IntVar()
        self.cascades_min_size_x_var.set(100)
        self.cascades_min_size_x = ttk.Combobox(master, textvariable=self.cascades_min_size_x_var, state="readonly")
        self.cascades_min_size_x['values'] = range(0,301,10)
        self.cascades_min_size_x.grid(row=6, column=3, columnspan=1)

        self.cascades_min_size_y_label = Label(master, text="Minimum size of a face (y axis)")
        self.cascades_min_size_y_label.grid(row=7, column=2)
        
        self.cascades_min_size_y_var = IntVar()
        self.cascades_min_size_y_var.set(100)
        self.cascades_min_size_y = ttk.Combobox(master, textvariable=self.cascades_min_size_y_var, state="readonly")
        self.cascades_min_size_y['values'] = range(0,301,10)
        self.cascades_min_size_y.grid(row=7, column=3, columnspan=1)

        self.cascades_avg_time_per_frame_var = StringVar()
        self.cascades_avg_time_per_frame_text = Label(master, textvariable=self.cascades_avg_time_per_frame_var)
        self.cascades_avg_time_per_frame_text.grid(row=8, column=2, columnspan=2)
        
        self.cascades_avg_fps_var = StringVar()
        self.cascades_avg_fps_text = Label(master, textvariable=self.cascades_avg_fps_var)
        self.cascades_avg_fps_text.grid(row=9, column=2, columnspan=2)
        
        self.run_cascades_batton = Button(master, text="Run cascades detector",
                                            command=lambda: self.cascades_detect_faces(filename=self.filename,
                                                                                    model=self.model_cascades_var.get(),
                                                                                    scaleFactor=float(self.cascades_scale_factor_var.get()),
                                                                                    minNeighbors=self.cascades_min_neighbors_var.get(),
                                                                                    minSize=(self.cascades_min_size_x_var.get(), self.cascades_min_size_y_var.get()))
                                         )
        self.run_cascades_batton.grid(row=10, column=2, columnspan=2)
        
        self.interrupt_cascades_detector_label = Label(master, text="Type 'q' to interrupt cascades detector")
        self.interrupt_cascades_detector_label.grid(row=11, column=2, columnspan=2)

        # dlib
        self.model_type_dlib_label = Label(master, text="Dlib")
        self.model_type_dlib_label.grid(row=2, column=5, columnspan=2)
        
        self.model_dlib_var = StringVar()
        self.model_dlib_var.set("hog")
        self.model_dlib = ttk.Combobox(master, textvariable=self.model_dlib_var, state="readonly")
        self.model_dlib['values'] = ("hog",
                                "cnn")
        self.model_dlib.grid(row=3, column=5, columnspan=2)
        
        self.dlib_unsample_number_label = Label(master, text="Number of times to unsample")
        self.dlib_unsample_number_label.grid(row=4, column=5)
        
        self.dlib_unsample_number_var = IntVar()
        self.dlib_unsample_number_var.set(1)
        self.dlib_unsample_number = ttk.Combobox(master, textvariable=self.dlib_unsample_number_var, state="readonly")
        self.dlib_unsample_number['values'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.dlib_unsample_number.grid(row=4, column=6)

        self.dlib_avg_time_per_frame_var = StringVar()
        self.dlib_avg_time_per_frame_text = Label(master, textvariable=self.dlib_avg_time_per_frame_var)
        self.dlib_avg_time_per_frame_text.grid(row=8, column=5, columnspan=2)
        
        self.dlib_avg_fps_var = StringVar()
        self.dlib_avg_fps_text = Label(master, textvariable=self.dlib_avg_fps_var)
        self.dlib_avg_fps_text.grid(row=9, column=5, columnspan=2)
        
        self.run_dlib_batton = Button(master, text="Run dlib detector",
                                        command=lambda: self.dlib_detect_faces(filename=self.filename,
                                                                            model=self.model_dlib_var.get(),
                                                                            number_of_times_to_upsample=self.dlib_unsample_number_var.get())
                                      )
        self.run_dlib_batton.grid(row=10, column=5, columnspan=2)

        self.interrupt_dlib_detector_label = Label(master, text="Type 'q' to interrupt dlib detector")
        self.interrupt_dlib_detector_label.grid(row=11, column=5, columnspan=2)

        # caffe
        self.model_type_caffe_label = Label(master, text="Caffe")
        self.model_type_caffe_label.grid(row=2, column=8, columnspan=2)
        
        self.model_caffe_var = StringVar()
        self.model_caffe_var.set("res10_300x300_ssd_iter_140000")
        self.model_caffe = ttk.Combobox(master, textvariable=self.model_caffe_var, state="readonly", width=27)
        self.model_caffe['values'] = ("res10_300x300_ssd_iter_140000")
        self.model_caffe.grid(row=3, column=8, columnspan=2)
        
        self.caffe_threshold_label = Label(master, text="Threshold")
        self.caffe_threshold_label.grid(row=4, column=8)
        
        self.caffe_threshold_var = StringVar()
        self.caffe_threshold_var.set(str(0.5))
        self.caffe_threshold = ttk.Combobox(master, textvariable=self.caffe_threshold_var, state="readonly")
        self.caffe_threshold['values'] = (str(0.1), str(0.2), str(0.3), str(0.4), str(0.5), str(0.6), str(0.7), str(0.8), str(0.9), str(0.95))
        self.caffe_threshold.grid(row=4, column=9, columnspan=1)
        
        self.caffe_avg_time_per_frame_var = StringVar()
        self.caffe_avg_time_per_frame_text = Label(master, textvariable=self.caffe_avg_time_per_frame_var)
        self.caffe_avg_time_per_frame_text.grid(row=8, column=8, columnspan=2)
        
        self.caffe_avg_fps_var = StringVar()
        self.caffe_avg_fps_text = Label(master, textvariable=self.caffe_avg_fps_var)
        self.caffe_avg_fps_text.grid(row=9, column=8, columnspan=2)
        
        self.run_caffe_batton = Button(master, text="Run caffe detector",
                                        command=lambda: self.caffe_detect_faces(filename=self.filename,
                                                                            threshold=float(self.caffe_threshold_var.get()))
                                       )
        self.run_caffe_batton.grid(row=10, column=8, columnspan=2)

        self.interrupt_caffe_detector_label = Label(master, text="Type 'q' to interrupt caffe detector")
        self.interrupt_caffe_detector_label.grid(row=11, column=8, columnspan=2)

    def cascades_detect_faces(self,
                              model="haarcascade_frontalface_default",
                              filename="mcem0_head.mpg",
                              scaleFactor=1.1,
                              minNeighbors=3,
                              minSize=(0, 0),
                              sleep_time=2):
        
        path = os.path.join(os.path.dirname(__file__), "models", "cascades", model + '.xml')
        detector = cv2.CascadeClassifier(path)
        
        results = np.empty([0, 2])
        
        cap = cv2.VideoCapture(filename)
        
        while(True):
            ret, img = cap.read()
            if not ret:
                break
            start_time = time.time()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(image=gray_img,
                                              scaleFactor=scaleFactor,
                                              minNeighbors=minNeighbors,
                                              minSize=minSize)
            detection_time = time.time() - start_time
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img, model, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(img, " {:.3f}".format(detection_time) + " sec", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            results = np.append(results, [[len(faces), detection_time]], axis=0)
            
            cv2.imshow(model,img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        avg_time_per_frame = results[:,1].mean()
        avg_fps = 1 / avg_time_per_frame
        self.cascades_avg_time_per_frame_var.set(str(round(avg_time_per_frame*1000, ndigits=2))+" ms")
        self.cascades_avg_fps_var.set(str(round(avg_fps, ndigits=2)))
        
        time.sleep(sleep_time)
        cap.release()
        cv2.destroyAllWindows()
    
    def dlib_detect_faces(self,
                          model="hog",
                          filename="mcem0_head.mpg",
                          number_of_times_to_upsample=1,
                          sleep_time=2):
        
        if model == "cnn":
            path = os.path.join(os.path.dirname(__file__), "models", "dlib", "mmod_human_face_detector.dat")
            detector = dlib.cnn_face_detection_model_v1(path)
        else:
            detector = dlib.get_frontal_face_detector()
        
        results = np.empty([0, 2])
        
        cap = cv2.VideoCapture(filename)
        
        while(True):
            ret, img = cap.read()
            if not ret:
                break
            start_time = time.time()
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_img, number_of_times_to_upsample)
            detection_time = time.time() - start_time
            for i, face in enumerate(faces):
                if model == "cnn":
                    face = face.rect
                x, y, w, h = face.left(), face.top(), face.right() - face.left(), face.bottom() - face.top()
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img, model, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(img, " {:.3f}".format(detection_time) + " sec", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            results = np.append(results, [[len(faces), detection_time]], axis=0)
            
            cv2.imshow(model,img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        avg_time_per_frame = results[:,1].mean()
        avg_fps = 1 / avg_time_per_frame
        self.dlib_avg_time_per_frame_var.set(str(round(avg_time_per_frame*1000, ndigits=2))+" ms")
        self.dlib_avg_fps_var.set(str(round(avg_fps, ndigits=2)))
        
        time.sleep(sleep_time)
        cap.release()
        cv2.destroyAllWindows()
        
    def caffe_detect_faces(self,
                           filename="mcem0_head.mpg",
                           threshold=0.5,
                           sleep_time=2):
        
        detector = cv2.dnn.readNetFromCaffe(os.path.join(os.path.dirname(__file__), "models", "caffe", "deploy.prototxt"),
                                            os.path.join(os.path.dirname(__file__), "models", "caffe", "res10_300x300_ssd_iter_140000.caffemodel"))
        
        results = np.empty([0, 2])
        
        cap = cv2.VideoCapture(filename)
        
        while(True):
            ret, img = cap.read()
            if not ret:
                break
            start_time = time.time()
            
            # grab the frame dimensions and convert it to a blob
            (h, w) = img.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network and obtain the detections and
            # predictions
            detector.setInput(blob)
            detections = detector.forward()
            
            detection_time = time.time() - start_time

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence < threshold:
                    continue

                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                x, y, w, h = startX, startY, endX - startX, endY - startY
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(img, "caffe:" + "{:.2f}%".format(confidence * 100), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                cv2.putText(img, " {:.3f}".format(detection_time) + " sec", (x,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            results = np.append(results, [[detections.shape[2], detection_time]], axis=0)
            
            cv2.imshow("caffe",img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        avg_time_per_frame = results[:,1].mean()
        avg_fps = 1 / avg_time_per_frame
        self.caffe_avg_time_per_frame_var.set(str(round(avg_time_per_frame*1000, ndigits=2))+" ms")
        self.caffe_avg_fps_var.set(str(round(avg_fps, ndigits=2)))
        
        time.sleep(sleep_time)
        cap.release()
        cv2.destroyAllWindows()
        
        

root = Tk()
my_gui = FaceDetector(root)
root.mainloop()