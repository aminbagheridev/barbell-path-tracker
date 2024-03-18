# -*- coding: utf-8 -*-
# author: marticles
import cv2
import dlib
import os
import numpy as np
from collections import deque

from flask import Flask, request, jsonify, Response, send_file
import tempfile

app = Flask(__name__)

class pathTracker(object):
    def __init__(self, videoName="default video"):
        self.video_size = (960, 540)
        self.box_color = (255, 255, 255)
        self.path_color = (0, 0, 255)
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'Dlib_Tracker', 'CamShift', 'Template_Matching']
        self.tracker_type = self.tracker_types[2]
        self.cap = cv2.VideoCapture(videoName)
        if not self.cap.isOpened():
            print("Video doesn't exist!", videoName)
            return
        self.frames_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.points = deque(maxlen=self.frames_count)

        # Initialize tracker based on the selected type
        if self.tracker_type == 'BOOSTING':
            self.tracker = cv2.TrackerBoosting_create()
        elif self.tracker_type == 'MIL':
            self.tracker = cv2.TrackerMIL_create()
        elif self.tracker_type == 'KCF':
            self.tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == 'TLD':
            self.tracker = cv2.TrackerTLD_create()
        elif self.tracker_type == 'MEDIANFLOW':
            self.tracker = cv2.TrackerMedianFlow_create()
        elif self.tracker_type == 'GOTURN':
            self.tracker = cv2.TrackerGOTURN_create()
        elif self.tracker_type == 'Dlib_Tracker':
            self.tracker = dlib.correlation_tracker()

    def onmouse(self,event, x, y, flags, param):
        """
        On mouse
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.track_window = None
        if self.drag_start:
            xmin = min(x, self.drag_start[0])
            ymin = min(y, self.drag_start[1])
            xmax = max(x, self.drag_start[0])
            ymax = max(y, self.drag_start[1])
            self.selection = (xmin, ymin, xmax, ymax)
        if event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            # box is set up here
            self.track_window = self.selection
            self.selection = None
          
    def drawing(self,image,x,y,w,h,timer):
        """
        Drawing the bound, center point and path for tracker in real-time
        """
        center_point_x = int(x + 0.5*w)
        center_point_y = int(y + 0.5*h)
        center = (center_point_x,center_point_y)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        self.points.appendleft(center)
        # tracker's bound
        cv2.rectangle(image, (int(x),int(y)), (int(x+w),int(y+h)), self.box_color, 2)
        # center point
        cv2.circle(image, center, 2, self.path_color, -1)
        # coordinate
        cv2.putText(image,"(X=" + str(center_point_x) + ",Y=" + str(center_point_y) + ")", (int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.path_color, 2)
        # fps
        cv2.putText(image,"FPS=" + str(int(fps)), (40,20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.path_color, 2)
        for i in range(1, len(self.points)):
            if self.points[i-1] is None or self.points[i] is None:
                continue
            # path of center point
            cv2.line(image, self.points[i-1], self.points[i], self.path_color,2)

    def start_tracking(self):
        i = 0
        last_image = None  # Initialize a variable to hold the last frame
        for f in range(self.frames_count):
            timer = cv2.getTickCount()
            ret, frame = self.cap.read()
            if not ret:
                print("End of video reached or error reading frame.")
                break
            print(f"Processing Frame {i}")

            image = cv2.resize(frame, self.video_size, interpolation=cv2.INTER_CUBIC)
            if i == 0:  # Assuming object selection for tracking is predefined or set up before this method is called
                pt1, pt2, pt3, pt4 = 310, 33, 587, 148
                bbox = (pt1, pt2, pt3 - pt1, pt4 - pt2)  # Define the initial bounding box
                if self.tracker_type == 'Dlib_Tracker':
                    self.tracker.start_track(image, dlib.rectangle(pt1, pt2, pt3, pt4))
                else:
                    self.tracker.init(image, bbox)

            if self.tracker_type == 'Dlib_Tracker':
                self.tracker.update(image)
                pos = self.tracker.get_position()
                x, y, w, h = int(pos.left()), int(pos.top()), int(pos.width()), int(pos.height())
            else:
                success, box = self.tracker.update(image)
                x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            cv2.rectangle(image, (x, y), (x + w, y + h), self.box_color, 2, 1)
            center = (int(x + 0.5 * w), int(y + 0.5 * h))
            self.points.appendleft(center)
            self.drawing(image, x, y, w, h, timer)

            # Update last_image with the current frame
            last_image = image.copy()

            i += 1

        if last_image is not None:
            # Ensure the 'Video' directory exists
            if not os.path.exists('Video'):
                os.makedirs('Video')
            # Save the last frame after processing all frames
            print('Saving the last frame...')
            cv2.imwrite(f"tracked_frame.jpg", last_image)
        else:
            print("No frames were processed.")

        self.cap.release()


@app.route('/track', methods=['POST'])
def track_barbell():
    # Check if the post request has the file part
    if 'video' not in request.files:
        return jsonify({"error": "No video part"}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No video selected"}), 400

    # Save the video to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    video.save(temp_video.name)

    # Initialize and run your tracker on the video
    tracker = pathTracker(videoName=temp_video.name)
    tracker.start_tracking()

    # Assuming your start_tracking method saves the last frame as 'track_result.jpg' in the 'Video' directory
    # Modify your pathTracker's start_tracking to ensure it does not depend on GUI functions and saves the last frame
    result_image_path = os.path.join('tracked_frame.jpg')
    if os.path.exists(result_image_path):
        return send_file(result_image_path, mimetype='image/jpeg')
    else:
        return jsonify({"error": "Failed to process video"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8080)
