#!/usr/bin/python
# coding=utf-8
import paramiko
import os


def upload(host, port, username, password, local, remote):
    check_local_dir(remote)
    upsf = paramiko.Transport((host, port))
    upsf.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(upsf)
    try:
        if os.path.isdir(local):  # 判断本地参数是目录还是文件
            print('开始上传文件夹：' + local)
            for f in os.listdir(local):  # 遍历本地目录
                print('开始上传文件：' + local + f + '  到  ' + remote + f)
                sftp.put(os.path.join(local + f), os.path.join(remote + f))  # 上传目录中的文件
        else:
            print('开始上传文件：' + local + '  到  ' + remote)
            sftp.put(local, remote)  # 上传文件
        print('上传完成!')
    except Exception as e:
        print('upload exception:', e)
    upsf.close()


def download(host, port, username, password, local, remote):
    check_local_dir(local)
    downsf = paramiko.Transport((host, port))
    downsf.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(downsf)
    try:
        if os.path.isdir(local):  # 判断本地参数是目录还是文件
            print('开始下载文件夹：' + remote)
            for f in sftp.listdir(remote):  # 遍历远程目录
                print('开始下载文件：' + remote + f + '  到  ' + local + f)
                sftp.get(os.path.join(remote + f), os.path.join(local + f))  # 下载目录中文件
        else:
            print('开始下载文件：' + remote + '  到  ' + local)
            sftp.get(remote, local)  # 下载文件
        print('下载完成!')
    except Exception as e:
        print('download exception:', e)
    downsf.close()


def check_local_dir(local_dir_name):
    """本地文件夹是否存在，不存在则创建"""
    if not os.path.exists(local_dir_name):
        os.makedirs(local_dir_name)


if __name__ == '__main__':
    host = '192.168.1.180'  # 主机
    up_host = '192.168.17.129'
    up_host1 = '192.168.1.110'
    port = 22  # 端口
    up_port = 20
    username = 'dji'  # 用户名
    password = 'dji'  # 密码
    username1 = 'ai'
    local = "G:/dji/flask-video-streaming/file_download/test/test/"  # 本地文件或目录，与远程一致，当前为windows目录格式，window目录中间需要使用双斜线

    remote = '/home/dji/Desktop/test2/'  # 远程文件或目录，与本地一致，当前为linux目录格式
    remote1 = '/home/ai/Desktop/test5/'
    remote2 = '/home/dji/Desktop/test2/'
    download(host, port, username, password, local, remote)  # 下载
    upload(up_host, port, username1, username1, local, remote1)  # 上传
