
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template, request, redirect
import threading
import argparse
import datetime
import imutils
import time
import cv2
import os
import pandas
from run_detection import Detector
import sys
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for



from utils.torch_utils import select_device, smart_inference_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

device_gl = select_device('')

detector = Detector(
		weights=ROOT / 'last.onnx',  # model path or triton URL
        # source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device=device_gl,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        # view_img=False,  # show results
        # save_txt=False,  # save results to *.txt
        # save_csv=False,  # save results in CSV format
        # save_conf=False,  # save confidences in --save-txt labels
        # save_crop=False,  # save cropped prediction boxes
        # nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        # augment=False,  # augmented inference
        # visualize=False,  # visualize features
        # update=False,  # update all models
        # project=ROOT / 'runs/detect',  # save results to project/name
        # name='exp',  # save results to project/name
        # exist_ok=False,  # existing project/name ok, do not increment
        # line_thickness=3,  # bounding box thickness (pixels)
        # hide_labels=False,  # hide labels
        # hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        # vid_stride=1,  # video frame-rate stride
        bs=None
)

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


UPLOAD_FOLDER = './test'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from flask import send_from_directory

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


if local_mode:
	vs = VideoStream(src=0).start()
else:
	cap.open(remote_ip)

time.sleep(2.0)

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

    # otherwise handle the GET request
    return '''
           <form method="POST">
               <div><label>RSTP video link: <input type="text" name="remote_ip_get"></label></div>
               <input type="submit" value="Submit">
           </form>'''
# ---------------------------------------------------------------------------------------------------

def detect_motion_core(frame, lock_en):

	global outputFrame, md, static_back, motion_list, time, df, motion

	# frame = imutils.resize(frame, width=400)
	# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# gray = cv2.GaussianBlur(gray, (5, 5), 0)

	# if static_back is None: 
	# 	static_back = gray
	# 	pass

	# diff_frame = cv2.absdiff(static_back, gray) 

	# thresh_frame = cv2.threshold(diff_frame, 15, 255, cv2.THRESH_BINARY)[1] 
	# thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2) 

	# cnts,_ = cv2.findContours(thresh_frame.copy(), 
	# 				cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

	# for contour in cnts: 
	# 	if cv2.contourArea(contour) >= 10000: 
	# 		motion = 1
	# 		(x, y, w, h) = cv2.boundingRect(contour) 
	# 		# making green rectangle around the moving object 
	# 		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3) 

	# motion_list.append(motion) 
	# outputFrame = frame.copy()

	# motion_list = motion_list[-2:] 
	outputFrame = frame.copy()
	if outputFrame is not None:
		outputFrame_pred = detector.detect(outputFrame)
		outputFrame = detector.get_img_with_preds(outputFrame, outputFrame_pred)


	# if lock_en:
	# 	with lock:
	# 		outputFrame = frame.copy()
	# else:
	# 	outputFrame = frame.copy()


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