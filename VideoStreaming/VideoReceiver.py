# ReceiveServer

import socket
import threading
import struct
import cv2
import numpy as np

receive_ip = '192.168.0.3'
receive_port = 5000
receive_resolution = (640,480)
receive_fps = 15
addr = (receive_ip, receive_port)

class camera_receive_set:
    def __init__(self):
        self.resolution = receive_resolution
        self.addr = addr
        self.src = 888+15
        self.interval = 0
        self.fps = receive_fps
    def set_socket(self):
        self.receiver = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    def socket_connect(self):
        self.set_socket()
        self.receiver.connect(self.addr)
        print("IP is %s:%d" % (self.addr[0],self.addr[1]))

    def decode_image(self):
        self.name = 'Camera from ' + self.addr[0]
        self.receiver.send(struct.pack("lhh", self.src, self.resolution[0], self.resolution[1]))
        while True:
            info = struct.unpack("lhh", self.receiver.recv(8))
            buf_size = info[0]
            if buf_size:
                try:
                    self.buf = b''
                    temp_buf = self.buf
                    while buf_size:
                        temp_buf = self.receiver.recv(buf_size)
                        buf_size -= len(temp_buf)
                        self.buf += temp_buf
                        data = np.fromstring(self.buf, dtype='uint8')
                        self.img = cv2.imdecode(data, 1)
                        cv2.imshow(self.name, self.img)
                except:
                    pass
                finally:
                    if cv2.waitKey(10)==27:
                        self.receiver.close()
                        cv2.destroyAllWindows()
                        break
    def get_data(self, interval):
        runThread = threading.Thread(target=self.decode_image)
        runThread.start()

    def run(self):
        self.addr = tuple(self.addr)
        self.socket_connect()
        self.get_data(self.interval)

def main():
    camera = camera_receive_set()
    camera.run()

if __name__ == '__main__':
    main()