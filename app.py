
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, request
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import pandas

vs = None                 
cap = cv2.VideoCapture()  
outputFrame = None        
lock = threading.Lock()   
frame_idx = 0            


bs_frame_count = int(os.environ.get('BS_FRAME_CNT', "32"))

multi_therad_en = False

remote_ip = os.environ.get('REMOTE_IP', "0.0.0.0")
local_mode = (remote_ip == "0.0.0.0")




static_back = None
motion = 0
motion_list = [ None, None ] 


df = pandas.DataFrame(columns = ["Start", "End"]) 


app = Flask(__name__)


if local_mode:
	vs = VideoStream(src=0).start()
else:
	cap.open(remote_ip)

time.sleep(2.0)

@app.route("/")
def index():
	""" Return the rendered template """

	return render_template("index.html")

@app.route('/change-rstp', methods=['GET', 'POST'])
def form_example():
    # handle the POST request
    global cap
    if request.method == 'POST':
        remote_ip_get = request.form.get('remote_ip_get')
        cap.release()
        cap = cv2.VideoCapture()
        cap.open(remote_ip_get)

        return '''
                  <h1>RSTP video link: {}</h1>'''.format(remote_ip_get)

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>RSTP video link: <input type="text" name="remote_ip_get"></label></div>
               <input type="submit" value="Submit">
           </form>'''
# ---------------------------------------------------------------------------------------------------

def detect_motion_core(frame, lock_en):

	global outputFrame, md, static_back, motion_list, time, df, motion

	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)

	if static_back is None: 
		static_back = gray
		pass

	diff_frame = cv2.absdiff(static_back, gray) 

	thresh_frame = cv2.threshold(diff_frame, 15, 255, cv2.THRESH_BINARY)[1] 
	thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 

	cnts,_ = cv2.findContours(thresh_frame.copy(), 
					cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

	for contour in cnts: 
		if cv2.contourArea(contour) >= 10000: 
			motion = 1
			(x, y, w, h) = cv2.boundingRect(contour) 
			# making green rectangle around the moving object 
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

	motion_list.append(motion) 

	motion_list = motion_list[-2:] 

	if lock_en:
		with lock:
			outputFrame = frame.copy()
	else:
		outputFrame = frame.copy()


def detect_motion(frame):
	global outputFrame, frame_idx

	detect_motion_core(frame, False)
	frame_idx += 1


def detect_motion_thread():
	global vs, cap, outputFrame, frame_idx, lock

	while True:

		ret = True
		if local_mode:
			frame = vs.read()
		else:
			ret, frame = cap.read()

		if ret:
			detect_motion_core(frame, True)
			frame_idx += 1


def generate():

	global outputFrame, cap, lock

	while True:

		with lock:

			if not multi_therad_en:
				ret = True
				if local_mode:
					frame = vs.read()
				else:
					ret, frame = cap.read()

				if ret:
					detect_motion(frame)

			if outputFrame is None:
				continue

			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			if not flag:
				continue

		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():

	return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--device_ip",
					type=str,
					default=os.environ.get('DEVICE_IP', "0.0.0.0"),
					help="ip address of the local device (0.0.0.0 means listen on all public IPs)")

	ap.add_argument("-o", "--server_port",
					type=int,
					default=int(os.environ.get('SERVER_PORT', "8000")),
					help="ephemeral port number of the server (1024 to 65535)")
	args = vars(ap.parse_args())

	if multi_therad_en:
		t = threading.Thread(target=detect_motion_thread)
		t.daemon = True
		t.start()

	app.run(host=args["device_ip"],
			port=args["server_port"],
			debug=True,
			threaded=True,
			use_reloader=False)

	if local_mode:
		vs.stop()
	else:
		cap.release()
