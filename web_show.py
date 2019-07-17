#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/7/2 17:49
#@Author: ykp
#@File  : web_show.py

from flask import Flask,render_template,Response
from face_recognition_web import Face_recognition
import threading
import cv2
import os
from datetime import timedelta
import tensorflow as tf

from pathlib import Path
from subprocess import run


global graph
graph = tf.get_default_graph()

camera = Face_recognition()

threads = []

app= Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=0)

# def resize():
#     ret,frame=cv2.imread('./static/test/face/original.input.png')
#     if ret is True:
#         enlarge = cv2.resize(frame, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite('./static/test/face/original.input.png')

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/face/')
def show():
    return render_template('face.html')

@app.route('/video_feed/')
def video_feed():
    return Response(gen(camera),mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(cam):
    while True:

        with graph.as_default():
            frame = cam.actual_time_recognition()
        if frame is None:
            break
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/top/')
def top():
    return render_template('top.html')
@app.route('/left/')
def left():
    return render_template('left.html')
@app.route('/main/')
def main():
    return render_template('main.html')
@app.route( '/cut/')
def cut():
    camera.save_last_frame()
    return render_template('cut.html')
@app.route('/original/')
def original():
    return render_template('original.html')
@app.route('/clarify/')
def clarify():
    flag=1
    try:
        t=threading.Thread(target = run_all)
        threads.append(t)
        t.start()
    except Exception:
        flag =  0
    return render_template('clarify.html')

def run_all():
    for param_file in Path('.').glob('test.json'):
        print(f'Run {param_file.stem}')
        run(['python', 'run.py', str(param_file)])

@app.route('/show_2/')
def show_2():
    return render_template('show_2.html')



if __name__ == '__main__':
    app.run(debug = True)