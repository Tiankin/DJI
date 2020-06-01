# SenderServer

import socket
import threading
import struct
import time
import cv2
import numpy as np

send_ip = '192.168.0.3'
send_port = 5000
send_resolution = (640,480)
send_fps = 15
addr = (send_ip, send_port)

class camera_send_set:
	def __init__(self):
		self.resolution = send_resolution
		self.fps = send_fps
		self.addr = addr
		self.set_socket(self.addr)
	def set_socket(self, addr):
		self.sender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
		self.sender.bind(addr)
		self.sender.listen(5)

def check(obj, send):
	info = struct.unpack('lhh', send.recv(8))

	if info[0]>888:
		obj.fps = int(info[0])-888
		obj.resolution = list(obj.resolution)
		obj.resolution[0] = info[1]
		obj.resolution[1] = info[2]
		obj.resolution = tuple(obj.resolution)
		return 1
	else:
		return 0

def encode_image(obj, send, addr):
	if check(obj,send)==0:
		return

	cap = cv2.VideoCapture(0)
	img_param = [int(cv2.IMWRITE_JPEG_QUALITY), obj.fps] # Set image frame
	while True:
		time.sleep(0.1)
		ret, obj.img = cap.read()
		obj.img = cv2.resize(obj.img, obj.resolution)
		ret, img_encode = cv2.imencode('.jpg', obj.img, img_param)
		img_arr = np.array(img_encode)
		obj.img_str = img_arr.tostring()

		try:
			send.send(struct.pack('lhh', len(obj.img_str), obj.resolution[0], obj.resolution[1])+obj.img_str)
		except:
			cap.release()
			return

def main():
	camera = camera_send_set()
	while True:
		send, send_addr = camera.sender.accept()
		senderThread = threading.Thread(None, target=encode_image, args=(camera, send, send_addr))
		senderThread.start()

if __name__ == '__main__':
	main()





























