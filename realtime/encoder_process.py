import os
import fcntl
import struct
import numpy as np
import time
from time import time_ns
from mpi4py import MPI
from threading import Thread, Timer, Event
from spykshrk.realtime import realtime_base, realtime_logging, binary_record, datatypes, main_process
from spykshrk.realtime.simulator import simulator_process

from spykshrk.realtime.datatypes import SpikePoint, LinearPosPoint, CameraModulePoint
from spykshrk.realtime.camera_process import VelocityCalculator, LinearPositionAssignment
from spykshrk.realtime.realtime_base import ChannelSelection, TurnOnDataStream, BinaryRecordSendComplete
from spykshrk.realtime.tetrode_models import kernel_encoder
import spykshrk.realtime.rst.RSTPython as RST
from spykshrk.realtime.trodes_data import TrodesNetworkDataReceiver
from spykshrk.realtime.gui_process import GuiEncoderParameterMessage

#timer to send uniform intensity function every deocder time bin if a decoded spike was not sent
class NoSpikeTimerThread(Thread):
    def __init__(self, event, spike_sent, mpi_send, timestamp, elec_grp_id, current_pos, config):
        Thread.__init__(self)
        self.stopped = event
        self.spike_sent = spike_sent
        self.mpi_send = mpi_send
        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.current_pos = current_pos
        self.config = config
        self.no_spike_counter = 1
        print('Spike timer on')

    def fetch_spike_sent(self,spike_sent):
        self.spike_sent = spike_sent
        return self.spike_sent

    def get_spike_info(self,timestamp,elec_grp_id,current_pos,config):
        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.current_pos = current_pos
        self.config = config
        self.pos_hist = np.ones(self.config['encoder']['position']['bins'])/self.config['encoder']['position']['bins']
        return self.timestamp, self.elec_grp_id, self.current_pos, self.pos_hist

    def run(self):
        #while not self.stopped.wait(self.config['pp_decoder']['bin_size']/30000):
        while not self.stopped.wait(0.01):
          self.fetch_spike_sent(self.spike_sent)
          #print('spike sent value from fetch function',self.spike_sent)
          self.get_spike_info(self.timestamp, self.elec_grp_id, self.current_pos, self.config)

          if self.spike_sent == True:
            self.spike_sent = False
            self.no_spike_counter = 1
            #print('spike sent')
          if self.spike_sent == False:
            self.mpi_send.send_decoded_spike(SpikeDecodeResultsMessage(timestamp=self.timestamp+(self.no_spike_counter*self.config['pp_decoder']['bin_size']),
                                                                      elec_grp_id=self.elec_grp_id,
                                                                      current_pos=self.current_pos,
                                                                      pos_hist=self.pos_hist))
            print('empty spike sent from',self.timestamp+(self.no_spike_counter*self.config['pp_decoder']['bin_size']),self.elec_grp_id,self.no_spike_counter)
            self.no_spike_counter += 1
            # hacky way to stop the timer, bascially after you hit pause in trodes the counter will ramp up and then the timer will stop
            if self.no_spike_counter > 1000:
              self.enc_man.stopFlag.set()
          else:
            print('initial spike from manager:',self.spike_sent)

##########################################################################
# Messages
##########################################################################
class SpikeDecodeResultsMessage(realtime_logging.PrintableMessage):

    _header_byte_fmt = '=qidqqi'
    _header_byte_len = struct.calcsize(_header_byte_fmt)

    def __init__(self, timestamp, elec_grp_id, current_pos, cred_int, pos_hist, send_time):
        self.timestamp = timestamp
        self.elec_grp_id = elec_grp_id
        self.current_pos = current_pos
        self.cred_int = cred_int
        self.pos_hist = pos_hist
        self.send_time = send_time

    def pack(self):
        pos_hist_len = len(self.pos_hist)
        pos_hist_byte_len = pos_hist_len * struct.calcsize('=d')


        message_bytes = struct.pack(self._header_byte_fmt,
                                    self.timestamp,
                                    self.elec_grp_id,
                                    self.current_pos,
                                    self.cred_int,
                                    self.send_time,
                                    pos_hist_byte_len)

        message_bytes = message_bytes + self.pos_hist.tobytes()

        return message_bytes

    @classmethod
    def unpack(cls, message_bytes):
        timestamp, elec_grp_id, current_pos, cred_int, send_time, pos_hist_len = struct.unpack(cls._header_byte_fmt,
                                                                    message_bytes[0:cls._header_byte_len])

        pos_hist = np.frombuffer(message_bytes[cls._header_byte_len:cls._header_byte_len+pos_hist_len])

        return cls(timestamp=timestamp, elec_grp_id=elec_grp_id,
                   current_pos=current_pos, cred_int=cred_int, pos_hist=pos_hist, send_time=send_time)


##########################################################################
# Interfaces
##########################################################################
class EncoderMPISendInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config):
        super(EncoderMPISendInterface, self).__init__(comm=comm, rank=rank, config=config)

    def send_record_register_messages(self, record_register_messages):
        self.class_log.debug("Sending binary record registration messages.")
        for message in record_register_messages:
            self.comm.send(obj=message, dest=self.config['rank']['supervisor'],
                           tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)
        self.comm.send(
            obj=BinaryRecordSendComplete(), dest=self.config['rank']['supervisor'],
            tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)
        self.class_log.debug("Done sending binary record registration messages.")

    def send_decoded_spike(self, query_result_message: SpikeDecodeResultsMessage):
        # decide which decoder to send spike to based on encoder rank
        if query_result_message.elec_grp_id in self.config['tetrode_split']['1st_half']:
            self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['decoder'][0],
                           tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)            
        elif query_result_message.elec_grp_id in self.config['tetrode_split']['2nd_half']:
            self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['decoder'][1],
                           tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA) 

        # original
        #self.comm.Send(buf=query_result_message.pack(), dest=self.config['rank']['decoder'],
        #               tag=realtime_base.MPIMessageTag.SPIKE_DECODE_DATA)

    def send_time_sync_report(self, time):
        self.comm.send(obj=realtime_base.TimeSyncReport(time),
                       dest=self.config['rank']['supervisor'],
                       tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE)

    def all_barrier(self):
        self.comm.Barrier()


class EncoderMPIRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, encoder_manager):
        super(EncoderMPIRecvInterface, self).__init__(comm=comm, rank=rank, config=config)
        self.enc_man = encoder_manager

        self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.process_request_message(msg)

            self.req = self.comm.irecv(tag=realtime_base.MPIMessageTag.COMMAND_MESSAGE.value)

    def process_request_message(self, message):

        if isinstance(message, realtime_base.TerminateMessage):
            self.class_log.debug("Received TerminateMessage")
            raise StopIteration()

        elif isinstance(message, realtime_base.NumTrodesMessage):
            self.class_log.debug("Received number of NTrodes Message.")
            self.enc_man.set_num_trodes(message)

        elif isinstance(message, ChannelSelection):
            self.class_log.debug("Received NTrode channel selection {:}.".format(message.ntrode_list))
            self.enc_man.select_ntrodes(message.ntrode_list)

        elif isinstance(message, TurnOnDataStream):
            self.class_log.debug("Turn on data stream")
            self.enc_man.turn_on_datastreams()

        elif isinstance(message, binary_record.BinaryRecordCreateMessage):
            self.enc_man.set_record_writer_from_message(message)

        elif isinstance(message, realtime_base.TimeSyncInit):
            self.enc_man.sync_time()

        elif isinstance(message, realtime_base.TimeSyncSetOffset):
            self.enc_man.update_offset(message.offset_time)

        elif isinstance(message, realtime_base.StartRecordMessage):
            self.enc_man.start_record_writing()

        elif isinstance(message, realtime_base.StopRecordMessage):
            self.enc_man.stop_record_writing()


class EncoderGuiRecvInterface(realtime_base.RealtimeMPIClass):
    def __init__(self, comm: MPI.Comm, rank, config, encoder_manager):
        super().__init__(comm=comm, rank=rank, config=config)
        self.encoder_manager = encoder_manager
        self.req = self.comm.irecv(source=self.config["rank"]["gui"])

    def __next__(self):
        rdy, msg = self.req.test()
        if rdy:
            self.encoder_manager.process_gui_request_message(msg)
            self.req = self.comm.irecv(source=self.config["rank"]["gui"])


##########################################################################
# Data handlers/managers
##########################################################################
class RStarEncoderManager(realtime_base.BinaryRecordBaseWithTiming):

    def __init__(self, rank, config, local_rec_manager, send_interface: EncoderMPISendInterface,
                 spike_interface: realtime_base.DataSourceReceiver,
                 pos_interface: realtime_base.DataSourceReceiver):

        super(RStarEncoderManager, self).__init__(rank=rank,
                                                  local_rec_manager=local_rec_manager,
                                                  send_interface=send_interface,
                                                  rec_ids=[realtime_base.RecordIDs.ENCODER_QUERY,
                                                           realtime_base.RecordIDs.ENCODER_OUTPUT],
                                                  rec_labels=[['timestamp',
                                                               'elec_grp_id',
                                                               'weight',
                                                               'position'],
                                                              ['timestamp',
                                                               'elec_grp_id','ch1','ch2','ch3','ch4',
                                                               'position','velocity','encode_spike',
                                                               'cred_int','decoder_num','neraby_spikes'] +
                                                              ['x{:0{dig}d}'.
                                                               format(x, dig=len(str(config['encoder']
                                                                                     ['position']['bins'])))
                                                               for x in range(config['encoder']['position']['bins'])]],
                                                  rec_formats=['qidd',
                                                               'qiddddddqqqq'+'d'*config['encoder']['position']['bins']])

        self.rank = rank
        self.config = config
        self.mpi_send = send_interface
        self.spike_interface = spike_interface
        self.pos_interface = pos_interface

        #initialize velocity calc and linear position assignment functions
        self.velCalc = VelocityCalculator(self.config)
        self.linPosAssign = LinearPositionAssignment(self.config)

        kernel = RST.kernel_param(mean=config['encoder']['mark_kernel']['mean'],
                                  stddev=config['encoder']['mark_kernel']['std'],
                                  min_val=config['encoder']['mark_kernel']['lower'],
                                  max_val=config['encoder']['mark_kernel']['upper'],
                                  interval=config['encoder']['mark_kernel']['interval'])

        pos_bin_struct = kernel_encoder.PosBinStruct([config['encoder']['position']['lower'],
                                                      config['encoder']['position']['upper']],
                                                     config['encoder']['position']['bins'])
        self.rst_param = kernel_encoder.RSTParameter(kernel, pos_bin_struct,
                                                     config['encoder']['position_kernel']['std'])
        self.encoders = {}

        self.spk_counter = 0
        self.pos_counter = 0
        self.count_noise_events = 0

        self.current_pos = 0
        self.current_vel = 0
        self.smooth_x = 0
        self.smooth_y = 0
        self.velocity_threshold = self.config['encoder']['vel']

        #initialize variables to record if a spike has been sent to decoder
        self.spike_sent = 3
        self.spike_timestamp = 0
        self.spike_elec_grp_id = 0
        self.decoder_number = 0
        self.encoding_spike = 0
        self.nearby_spikes = 0

        self.spxx = []
        self.crit_ind = 0

        # taskState variable for adding spikes to encoding model
        # set to 0 to test loading the old tree - will turn off new mark and use old occupancy in kernel_encoder
        self.taskState = 1

        time = MPI.Wtime()

        self.encoder_lat = np.zeros(200000)
        self.encoder_lat_ind = 0

        #start spike sent timer
        # NOTE: currently this is turned off because it increased the dropped spikes rather than decreased them
        # to turn on, uncomment the line, self.thread.start()
        # self.stopFlag = Event()
        # self.thread = NoSpikeTimerThread(self.stopFlag, self.spike_sent, self.mpi_send,
        #                                 self.spike_timestamp, self.spike_elec_grp_id, self.current_pos, self.config)
        #self.thread.start()


    def register_pos_datatype(self):
        # Register position, right now only one position channel is supported
        self.pos_interface.register_datatype_channel(-1)

    def set_num_trodes(self, message: realtime_base.NumTrodesMessage):
        self.num_ntrodes = message.num_ntrodes
        self.class_log.info('Set number of ntrodes: {:d}'.format(self.num_ntrodes))

    def select_ntrodes(self, ntrode_list):
        self.class_log.debug("Registering spiking channels: {:}.".format(ntrode_list))
        for ntrode in ntrode_list:
            #print('position parameters: ',self.rst_param)
            self.spike_interface.register_datatype_channel(channel=ntrode)

            # MEC: to reload old tree try changing True to False in this function call
            self.encoders.setdefault(ntrode, kernel_encoder.RSTKernelEncoder('/tmp/ntrode{:}'.
                                                                             format(ntrode),
                                                                             True, self.rst_param,self.config))

    def turn_on_datastreams(self):
        self.class_log.info("Turn on datastreams.")
        self.spike_interface.start_all_streams()
        self.pos_interface.start_all_streams()

    def process_next_data(self):

        time = MPI.Wtime()

        msgs = self.spike_interface.__next__()
        t0 = time_ns()
        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]
            if isinstance(datapoint, SpikePoint):
                #self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
                #                   datatype=datatypes.Datatypes.SPIKES, label='enc_recv')

                #print("new spike: ",datapoint.timestamp,datapoint.data)

                self.spk_counter += 1

                # first find dead channels from spykshrk config file and replace data with 0s (range: 0-3)
                if str(datapoint.elec_grp_id) in self.config['encoder']['dead_channels']:
                    #print('dead channel',datapoint.elec_grp_id)
                    dead_channel = self.config['encoder']['dead_channels'][str(datapoint.elec_grp_id)]
                    if self.spk_counter == 1:
                        print('tetrode:',datapoint.elec_grp_id,'dead channel:',dead_channel)                    
                    #print('channel is',dead_channel)
                    #print('data is',datapoint.data[dead_channel])
                    # zero out dead channel
                    datapoint.data[dead_channel] = 0
                    #print('zeroed data is',datapoint.data[dead_channel])
                #for key, value in self.config['encoder']['dead_channels'].items():
                #    print('dead channel list',key,value)
                #print('tet number',datapoint.elec_grp_id)

                # this line calculates the mark for each channel (max of the 40 voltage values)
                # NO. this does not!!
                # it should be the max on the highest channel and then the values of the other three in that bin
                # or this may simply always be bin 14 - but im not sure if the clip is the same as offline
                # original
                #old_amp_marks = [max(x) for x in datapoint.data]
                #print('old mark',old_amp_marks)
                #print('input voltage',datapoint.data)
                #print('mark',amp_marks[0],amp_marks[1],amp_marks[2],amp_marks[3])

                # new method - take 4 values from single bin with highest peak
                spike_data = datapoint.data
                channel_peaks = np.max(spike_data, axis=1)
                peak_channel_ind = np.argmax(channel_peaks)
                t_ind = np.argmax(spike_data[peak_channel_ind])
                amp_marks = spike_data[:, t_ind]
                #print('new mark',amp_marks)

                # test multiple dead channls - great it works!
                #if datapoint.elec_grp_id == 7:
                #    print('tet 7 marks:',amp_marks)

                #self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
                #                   datatype=datatypes.Datatypes.SPIKES, label='kde_start')
                
                # this looks up the current spike in the RStar Tree
                # also filter out large noise events (start with 1500uV)
                # turn off this filter because we remove duplicates in decoder process now
                #if not np.all(np.asarray(amp_marks)<self.config['encoder']['noise_max_amp']):
                #    print('large noise mark: positive')
                #    self.count_noise_events += 1
                #elif not np.all(np.asarray(amp_marks)>-self.config['encoder']['noise_max_amp']):
                #    print('large noise mark: negative')
                #    self.count_noise_events += 1
                #if (max(amp_marks) > self.config['encoder']['spk_amp'] and 
                #    np.all(np.asarray(amp_marks)<self.config['encoder']['noise_max_amp']) and
                #    np.all(np.asarray(amp_marks)>-self.config['encoder']['noise_max_amp'])):
                if max(amp_marks) > self.config['encoder']['spk_amp']:
                    #print(datapoint.timestamp,datapoint.elec_grp_id, amp_marks)
                    query_result = self.encoders[datapoint.elec_grp_id].query_mark_hist(
                        amp_marks,datapoint.timestamp,
                        datapoint.elec_grp_id)          # type: kernel_encoder.RSTKernelEncoderQuery
                    #print('decoded spike',query_result.query_hist)
                    if query_result is not None:
                        self.nearby_spikes = query_result.nearby_spikes
                        #print('nearby spikes',query_result.nearby_spikes)

                        #self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
                        #            datatype=datatypes.Datatypes.SPIKES, label='kde_end')

                        # for weight, position in zip(query_result.query_weights, query_result.query_positions):
                        #     self.write_record(realtime_base.RecordIDs.ENCODER_QUERY,
                        #                       query_result.query_time,
                        #                       query_result.ntrode_id,
                        #                       weight, position)

                        # record which decoder is receiving spike
                        if datapoint.elec_grp_id in self.config['tetrode_split']['1st_half']:
                            self.decoder_number = self.config['rank']['decoder'][0]
                        elif datapoint.elec_grp_id in self.config['tetrode_split']['2nd_half']:
                            self.decoder_number = self.config['rank']['decoder'][1]

                        # record if this is an encoding spike
                        if abs(self.current_vel) >= self.velocity_threshold and self.taskState == 1:
                            self.encoding_spike = 1
                        else:
                            self.encoding_spike = 0

                        # add credible interval here and add to message
                        # for this we will use 50% not 95%
                        self.spxx = np.sort(query_result.query_hist)[::-1]
                        self.crit_ind = (np.nonzero(np.diff(np.cumsum(self.spxx) >= 0.5, prepend=False))[0] + 1)[0]
                        # okay - this seems to be accurate
                        #if datapoint.elec_grp_id == 2:
                        #    print(self.crit_ind)
                        #    print(query_result.query_hist)


                        self.mpi_send.send_decoded_spike(SpikeDecodeResultsMessage(timestamp=query_result.query_time,
                                                                                elec_grp_id=query_result.elec_grp_id,
                                                                                current_pos=self.current_pos,
                                                                                cred_int=self.crit_ind,
                                                                                pos_hist=query_result.query_hist,
                                                                                send_time=time_ns()))
                        t1 = time_ns()
                        if self.encoder_lat_ind == self.encoder_lat.shape[0]:
                            self.encoder_lat = np.hstack((self.encoder_lat, np.zeros(self.encoder_lat.shape[0])))
                        self.encoder_lat[self.encoder_lat_ind] = (t1 - t0)
                        self.encoder_lat_ind += 1
                        #print('decode sent_from manager: ',query_result.query_time,query_result.elec_grp_id)

                        # save query_weights instead of query_hist: this will save number of spikes in rectangle
                        self.write_record(realtime_base.RecordIDs.ENCODER_OUTPUT,
                                        query_result.query_time,
                                        query_result.elec_grp_id,
                                        amp_marks[0],amp_marks[1],amp_marks[2],amp_marks[3],
                                        self.current_pos,self.current_vel,self.encoding_spike,
                                        self.crit_ind,self.decoder_number,self.nearby_spikes,
                                        *query_result.query_hist)
                        #print('query hist shape',query_result.query_hist.shape)

                        # weights have constantly changing size and are very small numbers - how to get # marks??
                        #print('weights',query_result.query_weights)

                        #self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=datapoint.elec_grp_id,
                        #                   datatype=datatypes.Datatypes.SPIKES, label='spk_dec')

                        # update spike_sent variable to True each time a spike is actually sent to decoder
                        # self.spike_sent = True
                        # self.thread.fetch_spike_sent(self.spike_sent)
                        # self.thread.get_spike_info(datapoint.timestamp,datapoint.elec_grp_id,self.current_pos,self.config)
                        #print('spike_sent value from manager:',self.spike_sent)

                    else:
                        # in the future, do we want to write a record if received spike but didn't decode?
                        # (occurs when there aren't enough spikes within the n-cube surrounding the received spike)
                        self.write_record(realtime_base.RecordIDs.ENCODER_OUTPUT,
                                        datapoint.timestamp,
                                        datapoint.elec_grp_id,
                                        amp_marks[0],amp_marks[1],amp_marks[2],amp_marks[3],
                                        self.current_pos,self.current_vel,self.encoding_spike,
                                        self.crit_ind,self.decoder_number,0,
                                        *np.zeros(41))

                    # this adds the current spike to the R Star Tree
                    # to turn off adding spike, comment out "new_mark" below
                    # added switch from config file: taskState == 1 (only add spikes if in first cued trials)
                    # can also add a secondary spike amplitude filter here
                    #if abs(self.current_vel) >= self.config['encoder']['vel'] and max(amp_marks)>self.config['encoder']['spk_amp']+50:
                    if (abs(self.current_vel) >= self.velocity_threshold and self.taskState == 1
                        and not self.config['ripple_conditioning']['load_encoding']):
                        if self.spk_counter % 5000 == 0:
                            print('added',self.spk_counter,'spikes to tree in tet',datapoint.elec_grp_id)
                            #print('number of noise events detected:',self.count_noise_events)

                        self.encoders[datapoint.elec_grp_id].new_mark(amp_marks)

                        #self.record_timing(timestamp=datapoint.timestamp, elec_grp_id=
                        #        datapoint.elec_grp_id,datatype=datatypes.Datatypes.SPIKES, label='spk_enc')
                        pass

                if self.spk_counter % 10000 == 0:
                    self.class_log.debug('Received {} spikes.'.format(self.spk_counter))

        msgs = self.pos_interface.__next__()
        if msgs is None:
            # No data avaliable but datastreams are still running, continue polling
            pass
        else:
            datapoint = msgs[0]
            timing_msg = msgs[1]

            if isinstance(datapoint, LinearPosPoint):
                #print('linearpospoint x value: ',datapoint.x)
                self.pos_counter += 1

                self.current_pos = datapoint.x
                self.current_vel = datapoint.vel
                for encoder in self.encoders.values():
                    encoder.update_covariate(datapoint.x)

                if self.pos_counter % 1000 == 0:
                    self.class_log.info('Received {} pos datapoints.'.format(self.pos_counter))
                pass

            if isinstance(datapoint, CameraModulePoint):
                #NOTE (MEC, 9-1-19): we need to include encoding velocity when calling update_covariate
                self.pos_counter += 1

                # read taskstate.txt for taskState - needs to be reset manually at begin of session
                # 1 = first cued arm trials, 2 = content trials, 3 = second cued arm trials
                # lets try 15 instead of 60
                if self.pos_counter % 15 == 0:
                  #if self.vel_pos_counter % 1000 == 0:
                      #print('thresh_counter: ',self.thresh_counter)
                    with open('config/taskstate.txt', 'rb') as f:
                        fd = f.fileno()
                        fcntl.fcntl(fd, fcntl.F_SETFL, os.O_NONBLOCK)
                        f.seek(-2, os.SEEK_END)
                        while f.read(1) != b'\n':
                            f.seek(-2, os.SEEK_CUR)
                        self.taskState = int(f.readline().decode()[0:1])
                      #if self.rank == 10:
                      #    print('encoder taskstate =',self.taskState)

                # run positionassignment, pos smoothing, and velocity calculator functions
                # currently smooth position and velocity
                self.smooth_x = self.velCalc.smooth_x_position(datapoint.x)
                self.smooth_y = self.velCalc.smooth_y_position(datapoint.y)
                self.current_vel = self.velCalc.calculator_no_smooth(self.smooth_x, self.smooth_y)
                #self.current_vel = self.velCalc.calculator(self.smooth_x, self.smooth_y)
                self.current_pos = self.linPosAssign.assign_position(datapoint.segment, datapoint.position)
                #print('x smoothing: ',datapoint.x,self.smooth_x)
                #print('y smoothing: ',datapoint.y,self.smooth_y)

                #print('encoder linear position: ',self.current_pos, ' velocity: ',self.current_vel)
                #print('segment: ',datapoint.segment)

                for encoder in self.encoders.values():
                    #print('encoder side current vel: ',self.current_vel)
                    encoder.update_covariate(self.current_pos,self.current_vel,self.taskState)

                if self.pos_counter % 5000 == 0:
                    self.class_log.info('Received {} pos datapoints.'.format(self.pos_counter))
    
    def save_data(self):
        lat_ms = self.encoder_lat[:self.encoder_lat_ind] / 1e6
        filepath = os.path.join(
            self.config['files']['output_dir'],
            self.config['files']['prefix'] + f'_encoder_lat_ms.{self.rank:02d}')
        np.save(filepath, lat_ms)
        self.class_log.debug("Wrote latencies to file")
    
    def process_gui_request_message(self, message):
        if isinstance(message, GuiEncoderParameterMessage):
            self.velocity_threshold = message.encoding_velocity_threshold
        else:
            self.class_log.debug(f"Received message of unknown type {type(message)}, ignoring")


##########################################################################
# Process
##########################################################################
class EncoderProcess(realtime_base.RealtimeProcess):
    def __init__(self, comm: MPI.Comm, rank, config):

        super().__init__(comm, rank, config)

        print('starting local record manager for rank',rank)
        self.local_rec_manager = binary_record.RemoteBinaryRecordsManager(manager_label='state', local_rank=rank,
                                                                          manager_rank=config['rank']['supervisor'])
        print('starting mpisend for rank',rank)
        self.mpi_send = EncoderMPISendInterface(comm=comm, rank=rank, config=config)

        if self.config['datasource'] == 'simulator':
            spike_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                        rank=self.rank,
                                                                        config=self.config,
                                                                        datatype=datatypes.Datatypes.SPIKES)

            pos_interface = simulator_process.SimulatorRemoteReceiver(comm=self.comm,
                                                                      rank=self.rank,
                                                                      config=self.config,
                                                                      datatype=datatypes.Datatypes.LINEAR_POSITION)
        elif self.config['datasource'] == 'trodes':
            print('about to configure network for decoding tetrode: ',self.rank)
            #time.sleep(10+2*self.rank)
            # spike_interface = simulator_process.TrodesDataReceiver(comm=self.comm,
            #                                                             rank=self.rank,
            #                                                             config=self.config,
            #                                                             datatype=datatypes.Datatypes.SPIKES)

            # pos_interface = simulator_process.TrodesDataReceiver(comm=self.comm,
            #                                                           rank=self.rank,
            #                                                           config=self.config,
            #                                                           datatype=datatypes.Datatypes.LINEAR_POSITION)

            spike_interface = TrodesNetworkDataReceiver(comm, rank, config, datatypes.Datatypes.SPIKES)
            pos_interface = TrodesNetworkDataReceiver(comm, rank, config, datatypes.Datatypes.LINEAR_POSITION)

            print('finished trodes setup for tetrode: ',self.rank)
        self.enc_man = RStarEncoderManager(rank=rank,
                                           config=config,
                                           local_rec_manager=self.local_rec_manager,
                                           send_interface=self.mpi_send,
                                           spike_interface=spike_interface,
                                           pos_interface=pos_interface)

        self.mpi_recv = EncoderMPIRecvInterface(comm=comm, rank=rank, config=config, encoder_manager=self.enc_man)

        # config['trodes_network']['networkobject'].registerTerminateCallback(self.trigger_termination)

        # First Barrier to finish setting up nodes
        self.class_log.debug("First Barrier")
        self.comm.Barrier()

    def main_loop(self):

        self.enc_man.setup_mpi()

        # First thing register pos datatype
        self.enc_man.register_pos_datatype()

        try:
            while True:
                self.mpi_recv.__next__()
                self.enc_man.process_next_data()

        except StopIteration as ex:
            self.class_log.info('Terminating EncodingProcess (rank: {:})'.format(self.rank))

        # self.enc_man.stopFlag.set()
        self.enc_man.save_data()
        self.class_log.info("Encoding Process reached end, exiting.")
