from pylsl import StreamInlet, resolve_stream
import time
import serial
import numpy as np

#com port connection
COMport="COM7"
ser=serial.Serial(COMport, 38400, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)

#look for stream
print("Looking for streams")
streams=resolve_stream('name', 'EventIDE_Signal_Stream')

#create inlet
print("test")
inlet=StreamInlet(streams[0])
print("Connected to 'EventIDE_Signal_Stream' stream")

#Read out stream
while True:
    print('test2`3')
    time.sleep(1)
    sample,timestamp=inlet.pull_chunk()
    if len(sample)==0:
        timestamp.append(0)
        sample.append(0)
    print('test3')
    print(f"Received sample at time {timestamp[0]}:")
    #[print(subsample) for subsample in sample]
    #bytesample=bytearray(sample)
    comsample=f"S{sample[0]}E"
    print(sample[0])
    ser.write(bytes(comsample, 'utf-8'))
