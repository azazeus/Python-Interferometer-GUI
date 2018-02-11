import time
import io
from functools import partial

from serial import Serial

class EnhancedSerial(Serial):
    def __init__(self, *args, **kwargs):
        #ensure that a reasonable timeout is set
        timeout = kwargs.get('timeout',0.1)
        if timeout < 0.01: timeout = 0.1
        kwargs['timeout'] = timeout
        Serial.__init__(self, *args, **kwargs)
        self.buf = ''
        
    def readline(self, maxsize=None, timeout=1):
        """maxsize is ignored, timeout in seconds is the max time that is way for a complete line"""
        tries = 0
        while 1:
            self.buf += self.read(512)
            pos = self.buf.find('\r\n')
            if pos >= 0:
                line, self.buf = self.buf[:pos+2], self.buf[pos+2:]
                return line
            tries += 1
            if tries * self.timeout > timeout:
                break
        line, self.buf = self.buf, ''
        return line

    def readlines(self, sizehint=None, timeout=1):
        """read all lines that are available. abort after timout
        when no more data arrives."""
        lines = []
        while 1:
            line = self.readline(timeout=timeout)
            if line:
                lines.append(line)
            if not line or line[-1:] != '\n':
                break
        return lines

class SMCError(Exception):
    pass 

class SMC(object):
    def __meta_injector(name, bases, dct):
        float_props = {'acceleration': 'AC', 'backlash_comp': 'BA', 'hysteresis_comp': 'BH', 'driver_voltage': 'DV', \
        'lowpass_freq': 'FD', 'following_error': 'FE', 'friction_comp': 'FF', 'jerk_time': 'JR', 'd_gain': 'KD', \
        'i_gain': 'KI', 'p_gain': 'KP', 'v_gain': 'KV', 'current_limits': 'QI', 'velocity': 'VA'}
        
        int_props = {'homesearch_type': 'HT'} 
        
        string_props = {'id': 'ID'}
        
        for k,v in float_props.iteritems():
            def _tmp(a, b):
                get_k = lambda self: float(self._ask(b, '?'))
                set_k = lambda self, x: self._write(b, x)
                dct.update({a: property(get_k, set_k)})
            _tmp(k,v)
            
        return type(name, bases, dct)

    __metaclass__ = __meta_injector
    
    def __init__(self, url=0, addr=1, debug=False):
        self.address = addr
        self._dev = EnhancedSerial(url, baudrate=57600, xonxoff=True, timeout=0.2)
        self.prev_cmd = None
        self.debug = debug
        
    def _debug(self, s):
        if self.debug is True:
            print(s), # changed to py3 style
            
    def _write(self, cc, arg=None):
        if isinstance(cc, basestring):
            self.prev_cmd = bytearray('{:d}{:s}'.format(self.address, cc))
            
            if arg is None:
                self._dev.write(bytearray('{:s}\r\n'.format(self.prev_cmd)))
            else:
                self._dev.write(bytearray('{:s}{}\r\n'.format(self.prev_cmd, arg)))
        else:
            raise NotImplementedError('Can only send string commands')
            
    def _ask(self, cc, arg=None):
        resp = None
        tries = 0        
        
        while True:
            try:
                self._write(cc, arg)
                self._debug('Loop #{:d}'.format(tries))
                line = self._dev.readline()
                resp = line.decode('ascii')
                if resp == u'':
                    continue
            except UnicodeError as e:
                self._debug('Trying again, resp is {}'.format(repr(resp)))
                tries += 1
                continue
            
            if tries > 4:
                raise SMCError('Ask didn\'t work')
                
            break
            
        if arg is None:
            self._debug('{:s} >> {:s}'.format(cc, resp.strip(' \r\n')))
        else:
            self._debug('{:s}{} >> {:s}'.format(cc, arg, resp.strip(' \r\n'))) 
           
        c = resp[3:]
        
        self._debug(' >> {:s}\n'.format(repr(c.strip(' \r\n'))))
        return c.strip(' \r\n')
                           
    def _get_error(self):
        resp = self._ask('TE')
        
        if resp == '@':
            return None
        else:
            raise SMCError(self._ask('TB{:s}'.format(resp)))
        
    def enter_configure(self):
        self._write('PW', 1)
    
    def load_esp(self, store=2):
        if store in xrange(1,3):
            self._write('ZX', store)
        
    def leave_configure(self):
        self._write('PW', 0)
        time.sleep(10)
    
    def reference(self):
        self._write('OR')
        
    def reset(self):
        self._write('RS')
        time.sleep(10)
        
    def disable(self):
        self._write('MM', 0)
    
    def enable(self):
        self._write('MM', 1)
        
    def get_state(self):
        resp = self._ask('TS')
        return resp
        
    def get_pos(self):
        pos = float(self._ask('TP'))
        pos_t = float(self._ask('TH'))
        #print '>> @{:6f} [{:6f}]'.format(pos, pos_t)
        return (pos, pos_t)
    
    def move(self, distance):
        if isinstance(distance, (float, int)):
            #resp = self._ask('PT', abs(float(distance)))
            self._write('PR', float(distance))
            #time.sleep(float(resp)*1.1)
            self.get_pos()
        else:
            raise NotImplementedError('cannot use {:s} as distance measure'.format(type(distance)))
            
    def move_to(self, distance):
        if isinstance(distance, (float, int)):
            pos,x = self.get_pos()
            
            self._write('PA', distance)
            while True:
                state = self.get_state()
                if state[4:] in ('28', '32', '33', '34'):
                    break
                elif int(state[0:4]) != 0:
                    self._get_error()
                    break
                
                #time.sleep(0.8)
        else:
            raise NotImplementedError('cannot use {:s} as distance measure'.format(type(distance)))
    
    def stop(self):
        self._write('ST')
            
    def get_state(self):
        return self._ask('TS')

    def close(self):
        self._dev.close()

