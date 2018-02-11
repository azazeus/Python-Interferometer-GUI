from __future__ import print_function
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from numpy.linalg import eig,inv
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxConstants import *
from antonSerial import *
import threading, time, logging,datetime
from math import *
from collections import deque
import h5py
from scipy.signal import decimate

########### Config Section
# store the type of input in a variable AI1_config
AI1_config=DAQmx_Val_PseudoDiff  # pseudo-differential

#specify max and min voltages the DAQ card will encounter
min_V=-10.0
max_V=10.0

# specify the units of above voltage numbers
AI1_units=DAQmx_Val_Volts   # Volts

#specify sampling rate
samp_rate=10000.0

#specify samples per channel
samp_per_chan=1000

# specify timeout in seconds
t_out=10.0

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )


class MainWindow(QtGui.QMainWindow):
    taskHandle=TaskHandle(0)
    def __init__(self, *args, **kwargs):
        QtGui.QMainWindow.__init__(self)
        self.create_main_window()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_timer)
    
    def is_stage_ready(self,e):
        while self.stage.get_state()[4:] != '33':
            #logging.debug('Position: %f', self.stage.get_pos()[0])
            time.sleep(1)
        self.stage.stop()
        self.e.set()
    
    ##### Set up the GUI in PyQT4
    def create_main_window(self):
        w = QtGui.QWidget()
        self.setCentralWidget(w)
        #### set window size and define layout containers 
        self.resize(1000,900)
        layout1 = QtGui.QGridLayout()
        layout2 = QtGui.QGridLayout()
        vboxlayout1= QtGui.QVBoxLayout()
        vboxlayout2= QtGui.QVBoxLayout()
        hbox1=QtGui.QHBoxLayout()
        
        ##### define various elements of the GUI
        self.start_loc=QtGui.QLineEdit('20.365')
        self.stop_loc=QtGui.QLineEdit('20.445')
        file_store_loc = 'M:\\data\\Y2014\\M12\\'
        self.file_loc=QtGui.QLineEdit(file_store_loc)
        self.file_comment_in=QtGui.QLineEdit("")
        self.file_comment_out=QtGui.QLineEdit("") 
        self.bA = QtGui.QPushButton('Start acquisition')
        self.bA.setSizePolicy(QtGui.QSizePolicy.Preferred,QtGui.QSizePolicy.Fixed)
        self.bA.clicked.connect(self.on_button_a)
        self.bB = QtGui.QPushButton('Stop acquisition')
        self.bB.setSizePolicy(QtGui.QSizePolicy.Preferred,QtGui.QSizePolicy.Fixed)
        self.bB.clicked.connect(self.on_button_b)
        self.openFile = QtGui.QPushButton('Open an existing HDF5 data file')
        self.openFile.clicked.connect(self.showDialog)
        self.plot1 = pg.PlotWidget(title="He-Ne monitor")
        self.plot1.resize(100,100)
        self.plot2 = pg.PlotWidget(title="PMT signal")
        self.plot3 = pg.PlotWidget(title="Interferogram") 
        self.plot1.setRange(xRange=(0,4), yRange=(0,4), padding=0.02, update=True, disableAutoRange=True)
        self.plot2.setRange(yRange=(-0.8,0.02),padding=0.02, update=True, disableAutoRange=True)
        #self.plot3.setRange(padding=0.02, update=True, disableAutoRange=True)
        self.plot4 = pg.PlotWidget(title="Spectrum") 
        self.label1=QtGui.QLabel("Enter start location in mm")
        self.label2=QtGui.QLabel("Enter stop location in mm")
        self.label3=QtGui.QLabel("Enter location to store file")
        self.label4=QtGui.QLabel("Enter file remarks if any")
        self.label5=QtGui.QLabel("File remarks")
        
        ##### layout the elements in the proper location
        layout1.addWidget(self.label1,0,0)
        layout1.addWidget(self.start_loc,0,1)
        layout1.addWidget(self.label2,1,0) 
        layout1.addWidget(self.stop_loc,1,1)
        layout2.addWidget(self.bA,0,0)
        layout2.addWidget(self.bB,0,1)
        vboxlayout1.addLayout(layout1)
        vboxlayout1.addWidget(self.label3) 
        vboxlayout1.addWidget(self.file_loc)
        vboxlayout1.addWidget(self.label4) 
        vboxlayout1.addWidget(self.file_comment_in)
        vboxlayout1.addLayout(layout2)
        vboxlayout1.addStretch()
        vboxlayout1.addWidget(self.openFile)
        vboxlayout1.addWidget(self.label5)
        vboxlayout1.addWidget(self.file_comment_out)
        hbox1.addWidget(self.plot1)
        hbox1.addWidget(self.plot2)
        hbox1.addStretch()
        hbox1.addLayout(vboxlayout1)
        vboxlayout2.addLayout(hbox1)
        vboxlayout2.addWidget(self.plot3)
        vboxlayout2.addWidget(self.plot4)
        w.setLayout(vboxlayout2)

        ##### generate handles to the plots for assigning data
        self.curve1 = self.plot1.plot(pen=None, symbol='o',symbolSize=1)
        self.curve2 = self.plot2.plot()
        self.curve3 = self.plot3.plot()
        self.curve4 = self.plot4.plot() 
        
        ##### define global variables for reading data from DAQ card
        self.read = int32()
        self.data = numpy.zeros((3 * samp_per_chan,), dtype=numpy.float64)

    #### dialog to open existing file
    def showDialog(self):
        fname=QtGui.QFileDialog.getOpenFileName(self, 'Open File', self.file_loc.text())
        #print("file name is: ",fname.type)
        f_rd_dat=h5py.File(str(fname), 'r')
        self.read_flag=True
        x=f_rd_dat['DAQ_raw_data/PD1_signal']
        y=f_rd_dat['DAQ_raw_data/PD2_signal']
        sig=f_rd_dat['DAQ_raw_data/PMT_signal']
        try:
            self.file_comment_out.setText(str(f_rd_dat['DAQ_raw_data'].attrs['comment']))
            print(f_rd_dat['DAQ_raw_data'].attrs['comment'])

        except:
            self.file_comment_out.setText("The remarks section is empty")
        dat_array=np.empty([3,x.size])
        dat_array[0]=x
        dat_array[1]=y
        dat_array[2]=sig
        self.ellipse_fit(dat_array)
    
    def on_button_a(self):
        
        self.bA.setEnabled(False)
        self.all_data = deque() 
        self.bA.setText('Running...')
        self.create_task_handle()
        try:
            self.stage=SMC(7,debug=False)
            motor_stat=self.stage.get_state()
        except:
            print ("Problem accessing the requested serial port.")
        if (motor_stat[4:] in ('32','33','34'))==False:
            print("Homing...wait")
            self.stage.enter_configure()
            self.stage.load_esp()
            self.stage.leave_configure()
            self.stage.reference()
        self.stage.velocity = 1.0
        self.stage.move_to(20)
        print("Ready!")
        while self.stage.get_state()[4:] == '28':
            pass    
        self.stage.move_to(float(self.start_loc.text()))
        while self.stage.get_state()[4:] == '28':
            pass
        self.stage.velocity=0.01
        self.stage.move_to(float(self.stop_loc.text()))
        self.e = threading.Event()
        self.stage_thread=threading.Thread(name='motor_check',target=self.is_stage_ready,args=(self.e,))
        self.stage_thread.start()
        self.timer.start(10)
        
    def on_button_b(self):
        self.timer.stop()
        self.shutdown()

    def shutdown(self):
        
        #plot_array=np.array(zip(*self.all_data))
        #self.curve1.setData(plot_array[0],plot_array[1])
        #self.curve2.setData(plot_array[2])
        
        if self.stage_thread.isAlive():
            print("Joining thread.")
            self.stage_thread.join()
        
        if Serial.isOpen(self.stage._dev) == True:
            print("Closing serial port")
            stage_stat=self.stage.close()
        #job of the task is done, delete it.
        try:
            self.bA.setEnabled(True)
            self.bA.setText('Start acquisition')
            daqstat=DAQmxClearTask(self.taskHandle)
        except:
            if daqstat != 0:
                print ("There was a problem shutting down the NI-DAQ device.")  
  
    def process_timer(self):
        if self.e.isSet() != True:
            DAQmxReadAnalogF64(self.taskHandle,samp_per_chan,t_out,DAQmx_Val_GroupByChannel,self.data,len(self.data),byref(self.read),None)
            data_sep=self.data.reshape((3, len(self.data)//3))
            self.all_data.append(self.data.copy())
            self.curve1.setData(data_sep[:2].T)
            self.curve2.setData(data_sep[2].T)
            #print(len(self.all_data))
        else:
            #self.analyze_this()
            self.timer.stop()
            self.data_collate()
            self.shutdown()
    
    def data_collate(self):
        tmp = np.empty((len(self.all_data), len(self.data)))
        for i, data in enumerate(self.all_data):
            tmp[i,:] = data
        print("tmp dimensions:", tmp.shape)
        tmp = tmp.reshape((i+1, 3, -1))    
        
        dat_array = np.einsum('ijk->jik', tmp).reshape(3,-1)
        self.read_flag=False
        self.write_dat(dat_array)
        self.ellipse_fit(dat_array)
        
    def ellipse_fit(self,dat_array):
        x=decimate(dat_array[0],20)
        y=decimate(dat_array[1],20)
        sig=decimate(dat_array[2],20)
          
        xlen=len(x)
        D=np.zeros((len(x),6)) #originally 20000 
        C=np.zeros((6,6))
        
        #pi=np.p
        c_vac=299792458

        # constraint matrix for 4ac-b^2 =1
        C[2,0]=2
        C[0,2]=2
        C[1,1]=-1
        # design matrix
        D[:,0]=np.square(x) # originally [1000:21000]
        D[:,1]=np.multiply(x,y)
        D[:,2]=np.square(y)
        D[:,3]=x
        D[:,4]=y
        D[:,5]=1
        # scattering matrix 
        S=np.dot(D.T,D)

        E,V=eig(np.dot(inv(S),C))
        eig_vec=V[:,np.argwhere(E>0)[0][0]]
        mu=np.sqrt(1/np.dot(eig_vec,np.dot(C,eig_vec.T)))
        a=mu*eig_vec
        
        # if np.square(a[1])-4*a[0]*a[2] == -1.0:
        print("Ellipse fit: b^2-4ac=",np.square(a[1])-4*a[0]*a[2])

        x_0=(a[2]*a[3]/2-a[1]*a[4]/4)/(np.square(a[1]/2)-a[0]*a[2])
        y_0=(a[0]*a[4]/2-a[1]*a[3]/4)/(np.square(a[1]/2)-a[0]*a[2])
        
        maj_axis=np.sqrt((2*(a[0]*(a[4]/2)**2+a[2]*(a[3]/2)**2+a[5]*(a[1]/2)**2-2*a[1]/2*a[3]/2*a[4]/2-a[0]*a[2]*a[5]))/((np.square(a[1]/2)-a[0]*a[2])*(np.sqrt((a[0]-a[2])**2+4*(a[1]/2)**2)-(a[0]+a[2]))))
        min_axis=np.sqrt((2*(a[0]*(a[4]/2)**2+a[2]*(a[3]/2)**2+a[5]*(a[1]/2)**2-2*a[1]/2*a[3]/2*a[4]/2-a[0]*a[2]*a[5]))/((np.square(a[1]/2)-a[0]*a[2])*(-np.sqrt((a[0]-a[2])**2+4*(a[1]/2)**2)-(a[0]+a[2]))))
        
        if a[0]>a[2]:
            ell_angle=1.0/2.0*np.arctan(a[1]/(a[0]-a[2]))
        elif a[2]>a[0]:
            ell_angle=pi+1.0/2.0*np.arctan(a[1]/(a[0]-a[2]))
        
        d_meas=np.vstack((x-x_0,y-y_0))
        rotation_matrix=[[cos(ell_angle),-sin(ell_angle)],[sin(ell_angle), cos(ell_angle)]]
        d_rot=np.dot(rotation_matrix,d_meas)
        phase_vec=np.arctan2(d_rot[1][:]*min_axis,d_rot[0][:]*maj_axis)
        
        p_val_old=phase_vec[0]
        time_vec=np.array([])
        phase_counter=0
        for p_val in phase_vec:
            if ((p_val_old>0) & (p_val <0)):
                phase_counter=phase_counter+1
                time_vec = np.append(time_vec,632.8e-9/(2*pi*c_vac)*(p_val+phase_counter*2*pi))
                p_val_old=p_val
            else:
                time_vec = np.append(time_vec,632.8e-9/(2*pi*c_vac)*(p_val+phase_counter*2*pi))
                p_val_old=p_val
        
        time_vec_unif=np.linspace(ceil(time_vec[0]*1e16),floor(time_vec[-1]*1e16),num=len(time_vec))*1e-16
        len(time_vec_unif)
        sig_interp=np.interp(time_vec_unif,time_vec,sig)
        
        sig_amp_f=np.fft.fft(-10*sig_interp)
        timestep=time_vec_unif[1]-time_vec_unif[0]
        #timestep
        freq_array=np.fft.fftfreq(sig.size, d=timestep)
        freq_array+=1
        wav_array=np.divide(c_vac,freq_array)
        #plot(freq_array,abs(sig_amp_f),'ro-')
        #plot(wav_array,abs(sig_amp_f),'ro-')
        #xlim([350e-9,500e-9])
        #xlim([c_vac/1100e-9,c_vac/650e-9])
        #ylim([0,20])
        eq_spect = np.divide(np.absolute(sig_amp_f),np.power(np.multiply(wav_array,1e6),3))
        
        self.curve3.setData(time_vec_unif,sig_interp)
        self.curve4.setData(freq_array, np.absolute(sig_amp_f))
    
    def write_dat(self,dat_array):
        if self.read_flag == False:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
            fname=str(self.file_loc.text())+ st + '.hdf5'

            with h5py.File(fname, 'w-') as f:
                raw_dat = f.create_group("DAQ_raw_data")
                raw_dat['PD1_signal'] = dat_array[0]
                raw_dat['PD2_signal'] = dat_array[1]
                raw_dat['PMT_signal'] = dat_array[2]
                raw_dat.attrs['comment'] = str(self.file_comment_in.text()) 
        #self.curve4.setData(wav_array,abs(eq_spect))

    def create_task_handle(self):
        # Specify which channels to add to the task
        ph_ch_name1="Dev1/ai0" # PD1
        ph_ch_name2="Dev1/ai1" # PD2
        ph_ch_name3="Dev1/ai2" # PMT
        #make some buffers to read in error values if any
        buf_size = 1000
        buf1 = ('\000' * buf_size)
        buf2 = ('\000' * buf_size)
        #create a new task using the task handle we made above
        try:
            cr_task_status=DAQmxCreateTask("", byref(self.taskHandle))
        except:
            #check if there is any error
            status=DAQmxGetErrorString(cr_task_status, buf1, buf_size)
            print("Error encountered in creating the task. Here's my best guess: ",buf1)
        #create read-in channels for the task
        try:
            cr_ch_status=DAQmxCreateAIVoltageChan (self.taskHandle, ph_ch_name1, "", AI1_config, min_V,max_V, AI1_units, None);
            cr_ch_status=DAQmxCreateAIVoltageChan (self.taskHandle, ph_ch_name2, "", AI1_config, min_V,max_V, AI1_units, None);
            cr_ch_status=DAQmxCreateAIVoltageChan (self.taskHandle, ph_ch_name3, "", AI1_config, min_V,max_V, AI1_units, None);
            cr_ch_status=DAQmxCfgSampClkTiming(self.taskHandle,"",samp_rate,DAQmx_Val_Rising,DAQmx_Val_ContSamps,samp_per_chan)
        except:
            status=DAQmxGetErrorString (cr_ch_status, buf2, buf_size)
            print("Error encountered in creating channels. Here's my best guess:", buf2) 


if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
