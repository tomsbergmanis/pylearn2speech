
__authors__ = "Pawel Swietojanski"
__copyright__ = "Copyright 2013, University of Edinburgh"
__credits__ = ["Pawel Swietojanski"]
__license__ = "3-clause BSD"
__maintainer__ = "Pawel Swietojanski"
__email__ = "p.swietojanski@ed.ac.uk"

import numpy
import struct
import random

from multiprocessing import Process, Queue, Pool

from string import Template
from StringIO import StringIO
from pylearn2.utils.speech_tmp import PathModifier
from pylearn2.datasets.speech_utils.providers import ListDataProvider, make_shell_call

def read_kaldi_matrix(buffer):    
       
    descr = struct.unpack('<xcccc', buffer.read(5)) #read 0B{F,D}{V,M,C}[space], function tested for 0BFM types only
      
    binary_mode = descr[0]
    repr_type = descr[1]
    cont_type = descr[2]      
    
    assert binary_mode=="B"
    
    if (repr_type=='F'):
        dtype = numpy.dtype(numpy.float32)
    elif (repr_type=='D'):
        dtype = numpy.dtype(numpy.float64)
    else:
        raise Exception('Wrong representation type in Kaldi header (is feats '
                        'compression enabled? - this is not supported in the '
                        'current version. Feel free to add this functionality.): %c'%(repr_type))

    rows,cols = 1, 1  
    if(cont_type=='M'):
        p1, rows = struct.unpack('<bi', buffer.read(5)) #bytes from 5 to 10
        p2, cols = struct.unpack('<bi', buffer.read(5)) #bytes from 10 to 15
        assert p1==4 and p2==4 #Number of bytes dimensionality is stored?         
    elif(cont_type=='V'):
        p1, rows = struct.unpack('<bi', buffer.read(5)) #bytes from 5 to 10
        assert p1==4
    else:
        raise Exception('Wrong container type in Kaldi header: %c'%(cont_type))
    
    assert rows > 0 and cols > 0 #just a range sanity checks
    assert rows < 360000 and cols < 30000 #just a sensible range sanity checks
    
    result = numpy.frombuffer(buffer.read(rows*cols*dtype.itemsize), dtype=dtype)
    if (cont_type=='M'):
        result = numpy.reshape(result, (rows, cols))
    
    return result

def read_uttid(buffer):
    uttid=''
    c=buffer.read(1)
    while c!=' ' and c!='':
       uttid+=c
       c=buffer.read(1)
    return uttid 

def read_ark_entry_from_buffer(buffer):
    """Reads a single Kaldi table archive entry and returns a tuple (uttid, ndarray)"""
    uttid = read_uttid(buffer)
    if uttid=='':
        return '' #sygnalize EOF or pipe
    mtrx = None
    try:
        mtrx = read_kaldi_matrix(buffer) #if this fails will raise (some) exception
    except Exception as e:
        raise e
    return (uttid, mtrx)

def read_ark_entry_from_archive(scp_entry):
    """Reads a single Kaldi table archive entry and returns a numpy array"""
    
    uttid, path = scp_entry.split(" ")
    ark_path, file_pos = path.split(":")
        
    try:
        ark = open(ark_path, 'rb')
        ark.seek(int(file_pos))
        feats = read_kaldi_matrix(ark)
        ark.close()
    except Exception as e:
        raise e
    
    return feats
    
def write_ark_entry_to_buffer(buffer, uttid, activations):
    activations = numpy.asarray(activations, dtype='float32')
    rows, cols = activations.shape
    buffer.write(struct.pack('<%ds'%(len(uttid)), uttid))
    buffer.write(struct.pack('<cxcccc', ' ','B','F','M',' '))
    buffer.write(struct.pack('<bi', 4, rows))
    buffer.write(struct.pack('<bi', 4, cols))
    buffer.write(activations)
    
    
class KaldiFeatsProviderUtt(ListDataProvider):

    def __init__(self, 
                 files_paths_list, 
                 template_shell_command=None, #"copy-feats-1file scp:\"echo ${SCP_ENTRY}|\"", 
                 randomize=False, 
                 max_utt=-1, 
                 max_time=-1,
                 path_modifier=PathModifier(), 
                 gmean=None, 
                 gstdev=None):
        try:
            super(KaldiFeatsProviderUtt, self).__init__(files_paths_list, path_modifier, gmean=gmean, gstdev=gstdev)
            self.max_utt = max_utt 
            self.randomize = randomize
            self.template_shell_command = (Template(template_shell_command) if (template_shell_command!=None) else None)
            if(self.randomize is True):
                random.shuffle(self.files_list)
            self._num_examples=8060414 #terrible hack for Herman's stuff
        except IOError as e:
            raise e
    
    def reset(self):
        self.index = 0
        if(self.randomize is True):
            random.shuffle(self.files_list)
    
    def __iter__(self):
        return self
    
    def next(self):
        if ((self.index >= self.list_size) or (self.max_utt>0 and self.index >= self.max_utt)):
            raise StopIteration
        
        utt_path = self.files_list[self.index]
        features=None
        try:
            if self.template_shell_command is None:
                features = read_ark_entry_from_archive(utt_path)
            else:
                features = self.shell_call(utt_path)
        except Exception as e:
            print 'Cannot load file: ', e
            
        self.index += 1
        print features.shape        
        return features, utt_path
    
    def shell_call(self, utt_path):
        
        shell_call_cmd = self.template_shell_command.substitute(SCP_ENTRY=utt_path)
        buffer_tuple = make_shell_call(shell_call_cmd)
        
        features = None
        if(buffer_tuple!=None):
            features = read_kaldi_matrix(StringIO(buffer_tuple[0]))
            #if self.index==0: #print the command only once (or Kaldi stderr)
            #    print buffer_tuple[1]
        else:
            raise Exception('Cannot extract a matrice using shell call %s .'%shell_call_cmd)
        
        return features
               
    @property
    def num_examples(self):
        return self._num_examples

 
class KaldiAlignFeatsProviderUtt(KaldiFeatsProviderUtt):

    def __init__(self, files_paths_list, align_file, template_shell_command=None, randomize=False,
                 max_utt=-1, max_time=-1, path_modifier=PathModifier(), gmean=None, gstdev=None, mapped=False):
        try:
            super(KaldiAlignFeatsProviderUtt, self).__init__(files_paths_list, template_shell_command,
                                                              randomize, max_utt, max_time, path_modifier, gmean=gmean, gstdev=gstdev)
            self.utt_skipped = 0
            self._num_classes = 0
            self._num_examples = 0
            
            self.files_info = {}
            for scp_entry in self.files_list:
                [utt, path] = scp_entry.split(' ', 1)
                self.files_info[utt] = scp_entry
                
            self.align_info = {}
            f = open(align_file, 'r')
            num_lines_align, num_aligns_found = 0, 0
            for line in f:
                num_lines_align += 1
                line = line.strip()
                if len(line) < 1:
                    continue  # remove empty lines
                [utt, alignment] = line.split(' ', 1)
                # skip empty alignments (these shouldn't be generated by Kaldi anyway)
                if len(alignment) < 1:
                    print 'Empty alignment for utterance %s'%utt
                    continue
                
                #only load the aligments which are actually needed
                if utt not in self.files_info:
                    continue
  
                self.align_info[utt] = numpy.loadtxt(StringIO(alignment), dtype=numpy.int32)

                tmp = self.align_info[utt].max()
                if tmp > self._num_classes:
                    self._num_classes = tmp
                self._num_examples += self.align_info[utt].shape[0]
                
                num_aligns_found += 1
                
                #read only 
                #if self.max_time>0 and self._num_examples > self.max_time:
                #    break;
                
                if len(self.align_info)==len(self.files_info):
                    break
                
                #if num_aligns_found >= max_utt and max_utt > 0:
                #    break
                
            f.close()
                      
            print 'KaldiAlignFeatsProviderUtt: Out of %d lines in %s, found %d alignments'%(num_lines_align, align_file, num_aligns_found)
            print 'KaldiAlignFeatsProviderUtt: %d feature files read'%len(self.files_list)
            print 'KaldiAlignFeatsProviderUtt: targets dimensionality is %i, number of examples %i'%(self._num_classes, self._num_examples)

            #when asked only subset of data, limit the lists here (given they are large enough at first place)
            if max_time > 0 and max_time < self._num_examples*0.01:
                print 'Limiting subset to %d seconds'%max_time
                self.randomize = False
                new_aligns_info = {}
                new_files_list = [];
                examples_loaded, idx = 0, 0;
                while examples_loaded*.01 < max_time: #Warning, assumed 10ms shift
                   scp_entry = self.files_list[idx]; idx += 1;
                   [utt, path] = scp_entry.split(' ', 1)
                   if utt not in self.align_info:
                       continue
                   new_files_list.append(scp_entry)
                   new_aligns_info[utt] = self.align_info[utt];
                   examples_loaded += new_aligns_info[utt].shape[0]
                   
                #alter some variables w.r.t new timing
                self.align_info = new_aligns_info
                self.files_list = new_files_list
                self.list_size = len(self.files_list)
                self._num_examples = examples_loaded #to make Trainer happy it showed all examples it supposed to

            self.mapped = mapped
        except IOError as e:
            raise e

    def __iter__(self):
        return self
    
    def reset(self):
        super(KaldiAlignFeatsProviderUtt, self).reset()
        self.utt_skipped = 0
    
    def next(self):
        if ((self.index >= self.list_size) or (self.max_utt>0 and self.index >= self.max_utt)):
            print 'KaldiAlignFeatsProviderUtt: Skipped %i utterances out of %i'%(self.utt_skipped, len(self.files_list))
            raise StopIteration

        utt_path = self.files_list[self.index]
        utt_id = utt_path.split(" ")[0]
        features, labels = None, None
        try:
            if self.template_shell_command is None:
                features = read_ark_entry_from_archive(utt_path)
            else:
                features = self.shell_call(utt_path)
        except Exception as e:
            print 'Cannot load file: ', str(e)

        self.index += 1
                
        if features is None:
            return (None, None), utt_path
          
        if utt_id in self.align_info:
            labels = self.align_info[utt_id]
        else:
            #print 'No alignments found for utterannce: %s\n\tIndex %d: %d alignments; %d feats'%(utt_id, self.index, len(self.align_info), len(self.files_list))
            self.utt_skipped += 1
            labels = None
            return (features, labels), utt_path
        
        #that shouldn't happen at this point
        if features.shape[0] != labels.shape[0]:
            print 'Alignments for %s have %i frames while the utt has %i. Skipping'%\
                    (utt_path, labels.shape[0], features.shape[0])
            return (features, None), utt_path

        if self.mapped==True:
            labels = labels-1

        return (features, labels), utt_path
    
    @property
    def num_classes(self):
        return self._num_classes
    
    @property
    def num_examples(self):
        return self._num_examples

class MultiStreamCall(Process):
    def __init__(self, thread_id, in_queue, out_queue):
        super(MultiStreamCall, self).__init__(name=thread_id)
        self.thread_id = thread_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        
    def run(self):
        #print 'Starting a %s'%self.thread_id
        scp_entry = self.in_queue.get(block=True)
        while (scp_entry is not None):
            try:
                features = read_ark_entry_from_archive(scp_entry)
                self.out_queue.put(features)
            except Exception as e:
                print e
                self.out_queue.put(None)
            scp_entry = self.in_queue.get(block=True)
        #print 'Finishing %s'%self.thread_id
        
        
class MultiStreamKaldiAlignFeatsProviderUtt(KaldiAlignFeatsProviderUtt):
    def __init__(self, files_paths_lists, align_file, template_shell_command, randomize=False,
                 subset=-1, mapped=False, concatenated=True, path_modifier=PathModifier()):
        
        assert files_paths_lists!=None and len(files_paths_lists)>1
        
        try:
            super(MultiStreamKaldiAlignFeatsProviderUtt, self).__init__(files_paths_lists[0], align_file, template_shell_command,
                                                              randomize=False, max_utt=subset, path_modifier=path_modifier, 
                                                              gmean=None, gstdev=None, mapped=mapped)
            
            self.concatenated=concatenated
            utt_paths = []
            #load the lists splitted into UTTIDs (keys) and PATHs into separate dictionaries
            for f in files_paths_lists:
                dp, dpd = ListDataProvider(f), {}
                for line in dp.files_list: 
                   [utt, path] = line.split(' ', 1)
                   dpd[utt]=line
                utt_paths.append(dpd)
            self.num_streams=len(utt_paths)
                 
            #now agree with aligns and get rid of any missing feats for any of the streams with the given UTTID
            self.utt_infos={} 
            for akey in self.align_info:
                paths, missing_paths = [], []
                for i in xrange(0, len(utt_paths)):
                    if akey in utt_paths[i]:
                        paths.append(utt_paths[i][akey])
                    else:
                        missing_paths.append(i)
                if len(missing_paths)==0:
                    self.utt_infos[akey]=paths
                else:
                    print 'UTTID %s - not found feats for the following stream IDs : %s'% \
                       (akey, ", ".join(str(x) for x in missing_paths))
            
            self.keys=self.utt_infos.keys()
            if randomize is True:
                random.shuffle(self.keys)
                
            #prepare (and run) stream providing threads
            self.stream_threads, self.stream_inqueues, self.stream_outqueues = [], [], []
            for s in xrange(0, self.num_streams):
                self.stream_inqueues.append(Queue(maxsize=1))
                self.stream_outqueues.append(Queue(maxsize=1))
                self.stream_threads.append(MultiStreamCall('Stream-%i'%s, self.stream_inqueues[s], self.stream_outqueues[s]))
                self.stream_threads[s].start()
                
        except IOError as e:
            raise e

    def __iter__(self):
        return self
    
    def reset(self):
       if self.randomize is True:
           random.shuffle(self.keys)
       self.index = 0
       
       for s in xrange(0, self.num_streams):
           self.stream_threads[s].terminate()
           
       self.stream_threads, self.stream_inqueues, self.stream_outqueues = [], [], []
       for s in xrange(0, self.num_streams):
           self.stream_inqueues.append(Queue(maxsize=1))
           self.stream_outqueues.append(Queue(maxsize=1))
           self.stream_threads.append(MultiStreamCall('Stream-%i'%s, self.stream_inqueues[s], self.stream_outqueues[s]))
           self.stream_threads[s].start()
        
    def next(self):
        if ((self.index >= len(self.utt_infos)) or (self.max_utt>0 and self.index >= self.max_utt)):
            for s in xrange(0, self.num_streams):
                self.stream_inqueues[s].put(None)
            raise StopIteration

        
        utt_id = self.keys[self.index]
        self.index += 1
        
        features, labels = None, None
        
        if utt_id in self.align_info:
            labels = self.align_info[utt_id]
        else:
            #print 'No alignments found for utterannce: %s\n\tIndex %d: %d alignments; %d feats'%(utt_id, self.index, len(self.align_info), len(self.files_list))
            return (features, labels), utt_id
        
        utt_paths = self.utt_infos[utt_id]
        feats_list = [None]*self.num_streams

        try:
            for s in xrange(0, self.num_streams):
                self.stream_inqueues[s].put(utt_paths[s])
            for s in xrange(0, self.num_streams): #this loop and queue secures proper synchronisation
                feats_list[s] = self.stream_outqueues[s].get(block=True)
        except Exception as e:
            print 'Something failed: ', e, buffer_tuple[1]  #, align_tuple[1]
            
        for i in xrange(0, len(feats_list)):
            if (feats_list[i] is None): 
                return (None, None), utt_id
               
        for feats in feats_list:
            if feats.shape[0] != labels.shape[0]:
                print 'Alignments for %s have %i frames while the utt has %i. Skipping'%\
                        (utt_path, labels.shape[0], features.shape[0])
                return (feats_list, None), utt_id
       
        if self.mapped==True:
            labels = labels-1
        
        if self.concatenated is False:
            return (feats_list, labels), utt_id
        
        features = numpy.concatenate(feats_list, axis=1) #concatenate the channels to make one big vector
                      
        return (features, labels), utt_id 
