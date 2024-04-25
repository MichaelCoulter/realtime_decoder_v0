from zmq import ZMQError
import numpy as np

from spykshrk.realtime import utils
from spykshrk.realtime.datatypes import Datatypes
from spykshrk.realtime.realtime_base import DataSourceReceiver
from spykshrk.realtime.datatypes import LFPPoint, SpikePoint, CameraModulePoint
from trodesnetwork.socket import SourceSubscriber

class TrodesNetworkDataReceiver(DataSourceReceiver):
    def __init__(self, comm, rank, config, datatype):
        if not datatype in (
            Datatypes.LFP,
            Datatypes.SPIKES,
            Datatypes.LINEAR_POSITION
        ):
            raise TypeError(f"Invalid datatype {datatype}")
        super().__init__(comm, rank, config, datatype)

        self.sub_obj = None
        
        self.start = False
        self.stop = False

        self.ntrode_ids = [] # only applicable for spikes and LFP
        self.inds_to_extract = None # only applicable for LFP
        self.ntrode_id_ind = 0 # only applicable for LFP
        self.scale_factor = self.config["trodes"]["voltage_scaling_factor"]

        self.temp_data = None

    def __next__(self):
        if self.stop:
            raise StopIteration()

        if not self.start:
            return None

        if self.datatype == Datatypes.LFP and self.is_subbed_multiple:
            # extracted all the channels from a single LFP packet so reset index.
            if self.ntrode_id_ind % self.n_subbed == 0:
                self.ntrode_id_ind = 0
            else:
                ind = self.inds_to_extract[self.ntrode_id_ind]
                ntid = self.ntrode_ids[self.ntrode_id_ind]
                datapoint = LFPPoint(
                    self.temp_data['localTimestamp'],
                    ind,
                    ntid,
                    self.temp_data['lfpData'][ind] * self.scale_factor)
                self.ntrode_id_ind += 1
                return datapoint, None
        
        try:
            self.temp_data = self.sub_obj.receive(noblock=True)
            
            if self.datatype == Datatypes.LFP:
                
                ind = self.inds_to_extract[self.ntrode_id_ind]
                ntid = self.ntrode_ids[self.ntrode_id_ind]
                datapoint = LFPPoint(
                    self.temp_data['localTimestamp'],
                    ind,
                    ntid,
                    self.temp_data['lfpData'][ind] * self.scale_factor)
                if self.is_subbed_multiple:
                    self.ntrode_id_ind += 1
                return datapoint, None

            elif self.datatype == Datatypes.SPIKES:
                
                ntid = self.temp_data['nTrodeId']
                if ntid in self.ntrode_ids:
                    datapoint = SpikePoint(
                        self.temp_data['localTimestamp'],
                        ntid,
                        np.array(self.temp_data['samples']) * self.scale_factor)
                    return datapoint, None
                else:
                    return None
            
            else:

                datapoint = CameraModulePoint(
                    self.temp_data['timestamp'],
                    self.temp_data['lineSegment'],
                    self.temp_data['posOnSegment'],
                    self.temp_data['x'],
                    self.temp_data['y'],
                    self.temp_data['x2'],
                    self.temp_data['y2'])
                return datapoint, None

        except ZMQError:
            return None

    def register_datatype_channel(self, channel):
        ntrode_id = channel
        if self.datatype in (Datatypes.LFP, Datatypes.SPIKES):
            if not ntrode_id in self.ntrode_ids:
                self.ntrode_ids.append(ntrode_id)
            else:
                self.class_log.debug(f"Already streaming from ntrode id {ntrode_id}")
        else:
            self.class_log.debug("Already set up to stream position, doing nothing")
            return
        
        if self.datatype == Datatypes.LFP:
            self.inds_to_extract = utils.get_ntrode_inds(self.config, self.ntrode_ids)

        self.class_log.debug(
            f"Set up to stream from ntrode ids {self.ntrode_ids}")

    def start_all_streams(self):
        if self.datatype == Datatypes.LFP:
            name = 'source.lfp'
        elif self.datatype == Datatypes.SPIKES:
            name = 'source.waveforms'
        else:
            name = 'source.position'

        server_address = utils.get_network_address(self.config)
        if server_address is None:
            self.sub_obj = SourceSubscriber(name)
        else:
            self.sub_obj = SourceSubscriber(name, server_address=server_address)

        self.start = True
        self.class_log.debug("Datastream activated")

    def stop_all_streams(self):
        self.start = False

    def stop_iterator(self):
        self.stop = True

    @property
    def n_subbed(self):
        return len(self.ntrode_ids)

    @property
    def is_subbed_multiple(self):
        return len(self.ntrode_ids) > 1
