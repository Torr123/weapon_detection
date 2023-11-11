import argparse
import os
import threading

import cv2
from flask import Flask
from flask import Response
from flask import flash, redirect, url_for
from flask import render_template, request
from flask import send_from_directory
from human_det.det_human import HumanDetector
from wd.stream import vs, cap, outputFrame, lock, frame_idx
from wd.stream import static_back, motion, motion_list
from wd.detector import detector
from wd.settings import local_mode
from utils.postproc import with_gun
from utils.metrics import box_iou

multi_therad_en = False

app = Flask(__name__)
humand_det = HumanDetector('yolov7-w6-pose.pt')

UPLOAD_FOLDER = './test'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'}

@app.route('/uploads/<name>')
def download_file(name):
	return send_from_directory(app.config["UPLOAD_FOLDER"], name)

def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
	global cap
	if request.method == 'POST':
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No selected file')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = file.filename#secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			cap.release()
			cap = cv2.VideoCapture()
			cap.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			return redirect(url_for('index', name=filename))
	return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route("/", methods=['GET', 'POST'])
def index():
	""" Return the rendered template """
	global cap
	if request.method == 'POST':
		remote_ip_get = request.form.get('remote_ip_get')
		cap.release()
		cap = cv2.VideoCapture()
		cap.open(remote_ip_get)
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

	return '''
           <form method="POST">
               <div><label>RSTP video link: <input type="text" name="remote_ip_get"></label></div>
               <input type="submit" value="Submit">
           </form>'''
# ---------------------------------------------------------------------------------------------------

def calc_dist(centr_1, centr_2):
	return ((centr_1[0]-centr_2[0])**2 + (centr_1[1]-centr_2[1])**2)**0.5

prev_people_with_gun = []

def detect_motion_core(frame, lock_en):
	global outputFrame, static_back, motion_list, motion, prev_people_with_gun
	# TODO Detect box and compare with previous frames
	outputFrame = frame.copy()
	if outputFrame is not None:
		outputFrame_pred = detector.detect(outputFrame)
		outputFrame = detector.get_img_with_preds(outputFrame, outputFrame_pred)

	det = humand_det.detect(outputFrame)
	det = with_gun(det, outputFrame_pred) # [[x, y, w, h]]
	# nimg = humand_det.get_img_with_preds(outputFrame, det)
	# nimg = humand_det.get_img_with_preds(outputFrame, outputFrame_pred)

	if len(prev_people_with_gun) == 0:
		pass
	if len(prev_people_with_gun) != 0 and len(det) != 0:
		print(prev_people_with_gun, det)
		ious = box_iou(prev_people_with_gun[:, :4], det[:, :4])
		_, idxs = ious.reshape(-1).sort(descending=True)
		new_people_with_gun = []
		prev_n, cur_n = ious.shape
		set_prev = set()
		set_cur = set()
		for idx in idxs:
			i = idx // cur_n
			j = idx % cur_n
			
			if i in set_prev or j in set_cur:
				continue

			set_prev.add(i)
			set_cur.add(j)
			
			det[j, -1] = 1 if det[j, -1] == 1 or prev_people_with_gun[i, -1] == 1 else 0

		
	prev_people_with_gun = det[det[:, -1] == 1] if len(det) != 0 else det

		
	nimg = humand_det.get_img_with_preds(outputFrame, det)


	return nimg


def detect_motion(frame):
	global outputFrame, frame_idx

	detect_motion_core(frame, False)
	frame_idx += 1


def detect_motion_thread():
	global vs, cap, outputFrame, frame_idx, lock
	print(f"detec")
	while True:
		ret = True
		if local_mode:
			frame = vs.read()
		else:
			ret, frame = cap.read()

		if ret:
			if frame_idx % 2 != 0:
				frame_idx += 1
				continue
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
