import numpy as np
import time
import struct
import queue
import ruamel.yaml
import socket
import threading
import cv2

class UDP_Manager:
    def __init__(self, callback, isServer = False, ip = '', port = 8083, frequency = 50, inet = 4):
        self.callback = callback
        
        self.isServer = isServer
        self.interval = 1.0 / frequency

        # self.available_addr = socket.getaddrinfo(socket.gethostname(), port)
        # self.hostname = socket.getfqdn(socket.gethostname())
        self.inet = inet
        self.af_inet = None
        self.ip = ip
        self.localIp = None
        self.port = port
        self.addr = (self.ip, self.port)
        self.running = False

        # self.serialNum = 0
        # self.recvPools = {} #{'IP:PORT': [{serialNum:[data..., recvNum, packetNum, timestamp]}]}

    def start(self):
        if self.inet == 4:
            self.af_inet = socket.AF_INET  # ipv4
            self.localIp = '127.0.0.1'
        elif self.inet == 6:
            self.af_inet = socket.AF_INET6 # ipv6
            self.localIp = '::1'
        self.sockUDP = socket.socket(self.af_inet, socket.SOCK_DGRAM)

        if self.isServer:
            self.roleName = 'Server'
        else:
            self.port = 0
            self.roleName = 'Client'
        
        self.sockUDP.bind((self.ip, self.port))
        self.sockUDP.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 212992)
        self.addr = self.sockUDP.getsockname()
        self.ip = self.addr[0]
        self.port = self.addr[1]
        print(self.roleName, '(UDP) at:', self.ip, ':', self.port)
        
        self.running = True
        self.thread = threading.Thread(target = self.receive, args=())
        self.thread.setDaemon(True)
        self.thread.start()  #打开收数据的线程
        
    # def ListAddr(self):
    #     for item in self.available_addr:
    #         if item[0] == self.af_inet:
    #             print(item[4])
    
    def receive(self):
        while self.running:
            time.sleep(self.interval)
            while self.running:
                try:
                    recvData, recvAddr = self.sockUDP.recvfrom(65535) #等待接受数据
                except:
                    break
                if not recvData:
                    break
                self.callback(recvData, recvAddr)
            
    def send(self, data, addr):
        self.sockUDP.sendto(data, addr)

    def close(self):
        self.running = False

class Sensor:
    def __init__(self, recvCallback = None, port = 9988, maxQSize = 5, callbackParam = None):
        '''
        Parameters
        ----------
        recvCallback: 回调函数
            recvCallback(frame)，接收一个参数frame，frame为触觉数据帧。
            在每次收到一个完整的Tac3D-Desktop传来的数据帧后，将自动调用
            一次该回调函数。回调函数的参数为所接收到的触觉数据帧。
            数据帧frame的数据结构介绍见Sensor.getFrame()的说明部分
        port: 整形
            接收Tac3D-Desktop传来的数据的UDP端口。需要注意此端口应与
            在Tac3D-Desktop中设置的数据接收端口一致。
        maxQSize: 整形
            最大接收队列长度。在使用Sensor.getFrame()获取触觉数据帧时，
            数据帧缓存队列的最大长度。（若使用recvCallback方法获取触觉
            数据帧，则不受此参数的影响）
        '''
        self._UDP = UDP_Manager(self._recvCallback_UDP, isServer = True, port = port)
        self._recvQueue = queue.Queue()
        self._recvBuffer = {}
        self._maxQSize = maxQSize
        self._recvCallback = recvCallback
        self._callbackParam = callbackParam
        self._count = 0
        self._yaml = ruamel.yaml.YAML()
        self._startTime = time.time()
        self._UDP.start()
        self._recvFlag = False
        self._fromAddrMap = {}
        self.frame = None
        
    def _recvCallback_UDP(self, data, addr):
        
        serialNum, pktNum, pktCount = struct.unpack('=IHH', data[0:8])
        currBuffer = self._recvBuffer.get(serialNum)
        if currBuffer is None:
            currBuffer = [0.0, pktNum, 0, [None]*(pktNum+1)]
            self._recvBuffer[serialNum] = currBuffer
        currBuffer[0] = time.time()
        currBuffer[2] += 1
        currBuffer[3][pktCount] = data[8:]
        
        if currBuffer[2] == currBuffer[1]+1:
            try:
                frame = self._decodeFrame(currBuffer[3][0], b''.join(currBuffer[3][1:]))
                initializeProgress = frame.get('InitializeProgress')
                if initializeProgress != None:
                    if initializeProgress != 100:
                        return
            except:
                return
            self.frame = frame
            self._fromAddrMap[frame['SN']] = addr
            self._recvQueue.put(frame)
            if self._recvQueue.qsize() > self._maxQSize: 
                self._recvQueue.get()
            self._recvFlag = True
            if not self._recvCallback is None:
                self._recvCallback(frame, self._callbackParam)
            del self._recvBuffer[serialNum]
        
        self._count += 1
        if self._count > 2000:
            self._cleanBuffer()
            self._count = 0
        
    def _decodeFrame(self, headBytes, dataBytes):
        #print(headBytes)
        #print(len(dataBytes))
        #print(headBytes.decode('ascii'))
        head = self._yaml.load(headBytes.decode('ascii'))
        frame = {}
        frame['index'] = head['index']
        frame['SN'] = head['SN']
        frame['sendTimestamp'] = head['timestamp']
        frame['recvTimestamp'] = time.time() - self._startTime
        for item in head['data']:
            dataType = item['type']
            if dataType == 'mat':
                dtype = item['dtype']
                if dtype == 'f64':
                    width = item['width']
                    height = item['height']
                    offset = item['offset']
                    length = item['length']
                    frame[item['name']] = np.frombuffer(dataBytes[offset:offset+length], dtype=np.float64).reshape([height, width])
            elif dataType == 'f64':
                offset = item['offset']
                length = item['length']
                frame[item['name']] = struct.unpack('d', dataBytes[offset:offset+length])[0]
            elif dataType == 'i32':
                offset = item['offset']
                length = item['length']
                frame[item['name']] = struct.unpack('i', dataBytes[offset:offset+length])[0]
            elif dataType == 'img':
                offset = item['offset']
                length = item['length']
                frame[item['name']] = cv2.imdecode(np.frombuffer(dataBytes[offset:offset+length], np.uint8), cv2.IMREAD_ANYCOLOR)
        return frame
        
    def _cleanBuffer(self, timeout = 5):
        currTime = time.time()
        delList = []
        for item in self._recvBuffer.items():
            if currTime - item[1][0] > timeout:
                delList.append(item[0])
        for item in delList:
            #print(self._recvBuffer[item][0:3])
            del self._recvBuffer[item]
        
    def getFrame(self):
        '''
        获取缓存数据帧队中的数据帧。
        Return
        ----------
        frame: 触觉数据帧
            当缓存数据帧队列不为空时，则返回队列最前端的帧。当缓存数据
            帧队列为空时则返回None
            frame的数据类型为字典
            {
                "SN": （字符串）传感器SN码——所接收到的数据帧来自的触
                    觉传感器的SN码，用于标记触觉数据帧的来源
                "index": （整形）帧序号——自每一个触觉传感器启动时从0
                    开始计数的帧编号。每个触觉传感器独立计数。但由于每
                    个触觉传感器在启动时有一段初期校准采样，触觉数据帧
                    的传输是从初期校准采样结束后开始的。故实际接收到的
                    触觉数据帧的序号并不是从0开始的。
                "sendTimestamp": （浮点数）数据发送时间戳——自每一个触
                    觉传感器在Tac3D-Desktop上启动时从0开始计算的触觉数
                    据帧发送时间。该时间由Tac3D-Desktop端计算。
                "recvTimestamp": （浮点数）数据接收时间戳——自每一个
                    Sensor实例初始化时从0开始计算的接收到触觉数据帧的时
                    间。该时间由Tac3D::Sensor端计算。
                "3D_Positions": （numpy.array）三维形貌——是一个形状为
                    400行3列的二维数组。数组的每一行分别对应一个标志点，
                    数组的每一列分别对应标志点的x、y、z坐标。
                "3D_Displacements": （numpy.array）三维位移场——是一个
                    形状为400行3列的二维数组。数组的每一行分别对应一个标
                    志点，数组的每一列分别对应标志点的x、y、z方向位移。
                "3D_Forces": （numpy.array）三维位移场——是一个形状为400
                    行3列的二维数组。数组的每一行分别对应一个标志点，数
                    组的每一列分别对应标志点附近区域的x、y、z方向受力。
            }
        '''
        if not self._recvQueue.empty():
            return self._recvQueue.get()
        else:
            return None
    
    def waitForFrame(self):
        '''
        阻塞等待接收数据帧，直到接收到第一帧数据帧为止。需注意，此函数的
        用途通常为等待Tac3D-Desktop启动传感器，而非在接收到一帧的数据后
        等待下一帧数据。若在调用此函数之前已经成功接收到来自Tac3D-Desktop
        的数据，则在此函数中不会进行任何阻塞等待。
        '''
        print('Waiting for Tac3D sensor...')
        self._recvFlag = False
        while not self._recvFlag:
            time.sleep(0.1)
        print('Tac3D sensor connected.')


    def calibrate(self, SN):
        '''
        向Tac3D-Desktop发送一次传感器校准信号，要求设置当前的3D变形状态为
        无变形是的状态以消除“温漂”或“光飘”的影响。相当于在Tac3D-Desktop
        端点击一次 “零位校准”按键的效果。
        '''
        addr = self._fromAddrMap.get(SN)
        if addr != None:
            print('Calibrate signal send to %s.' % SN)
            self._UDP.send(b'$C', addr)
        else:
            print("Calibtation failed! (sensor %s is not connected)" % SN)
            
    def quitSensor(self, SN):
        '''
        向Tac3D-Desktop发送一次结束传感器线程的信号。相当于在Tac3D-Desktop
        端点击一次 “关闭传感器”按键的效果。
        '''
        addr = self._fromAddrMap.get(SN)
        if addr != None:
            print('Quit signal send to %s.' % SN)
            self._UDP.send(b'$Q', addr)
        else:
            print("Quit failed! (sensor %s is not connected)" % SN)

if __name__ == '__main__':
    SN = ''
    idx = -1
    sendTimestamp = 0.0
    recvTimestamp = 0.0

    P, D, F, Fr, Mr = None, None, None, None, None
    
    def Tac3DRecvCallback(frame, param):
        global SN, idx, sendTimestamp, recvTimestamp, P, D, F, Fr, Mr
        # 获取SN
        SN = frame['SN']
        
        # 获取帧序号
        idx = frame['index']
        
        # 获取时间戳
        sendTimestamp = frame['sendTimestamp']
        recvTimestamp = frame['recvTimestamp']

        # 获取标志点三维形貌
        # P矩阵为400行3列，400行分别对应400个标志点，3列分别为各标志点的x，y和z坐标
        P = frame.get('3D_Positions')

        # 获取标志点三维位移场
        # D矩阵为400行3列，400行分别对应400个标志点，3列分别为各标志点的x，y和z位移
        D = frame.get('3D_Displacements')

        # 获取三维分布力
        # F矩阵为400行3列，400行分别对应400个标志点，3列分别为各标志点附近区域所受的x，y和z方向力
        F = frame.get('3D_Forces')
        
        # 获得三维合力
        # Fr矩阵为1x3矩阵，3列分别为x，y和z方向合力
        Fr = frame.get('3D_ResultantForce')

        # 获得三维合力矩
        # Mr矩阵为1x3矩阵，3列分别为x，y和z方向合力矩
        Mr = frame.get('3D_ResultantMoment')

    # 创建Sensor实例，设置回调函数为上面写好的Tac3DRecvCallback，设置UDP接收端口为9988
    tac3d = Sensor(recvCallback=Tac3DRecvCallback, port=9988)

    # 等待Tac3D-Desktop端启动传感器并建立连接
    tac3d.waitForFrame()
    
    time.sleep(5) # 5s

    # 发送一次校准信号（应确保校准时传感器未与任何物体接触！否则会输出错误的数据！）
    tac3d.calibrate(SN)

    time.sleep(5) #5s

    # 获取frame的另一种方式：通过getFrame获取缓存队列中的frame
    frame = tac3d.getFrame()
    if not frame is None:
        print(frame['SN'])

    time.sleep(5) #5s

    # # 发送一次关闭传感器的信号（不建议使用）
    # tac3d.quitSensor(SN)





