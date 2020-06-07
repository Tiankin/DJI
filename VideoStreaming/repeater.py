import paramiko as pr
import os

down_host_ip = ''
up_host_ip = ''
down_port = 5000
up_port = 8000
username = ''
password = ''
down_remote_path = ''
up_remote_path = ''
local_path = ''

def download(host_ip,port,username,password,remote_path,local_path):
    trans = pr.Transport((host_ip, port))  # set ssh connect
    trans.connect(username=username, password=password)  # bind username, pwd
    ssh = pr.SFTPClient.from_transport(trans)
    try:  # download file or dir
        if os.path.isdir(local_path): # check it is file or dir
            for file in ssh.lisrdir(remote_path):
                ssh.get(os.path.join(remote_path+file), os.path.join(local_path+file))
        else:
            ssh.get(remote_path, local_path)
        print('Download finished!')
    except Exception as e:
        print('Download error:', e)
        trans.close()

def upload(host_ip,port,username,password,remote_path,local_path):
    trans = pr.Transport((host_ip, port))
    trans.connect(username=username, password=password)
    ssh = pr.SFTPClient.from_transport(trans)
    try:  # upload file or dir
        if os.path.isdir(local_path): # check it is file or dir
            for file in os.listdir(local_path):
                ssh.put(os.path.join(local_path+file), os.path.join(remote_path+file))
        else:
            ssh.put(local_path,remote_path)
        print('Upload finished!')
    except Exception as e:
        print('Upload error!', e)
        trans.close()

def main():
    download(down_host_ip, down_port, username, password, down_remote_path, local_path)
    upload(up_host_ip, up_port, username, password, up_remote_path, local_path)

if __name__ == '__main__':
    main()