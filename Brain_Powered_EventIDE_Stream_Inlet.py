from pylsl import StreamInlet, resolve_stream
import time
import serial
import numpy as np
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
from skorch.callbacks import LRScheduler
import torch

# load model
lr = 0.0625 * 0.01
weight_decay = 0.8 * 0.001
batch_size = 64
n_epochs = 20
n_chans = 19
n_classes = 3
input_window_samples = 2501
cuda = torch.cuda.is_available()  # check if GPU is available
device = "cuda" if cuda else "cpu"
classes = [0, 1, 2]

model = ShallowFBCSPNet(
    in_chans=n_chans,
    n_classes=n_classes,
    input_window_samples=input_window_samples,
    final_conv_length="auto",
)

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
    max_epochs=n_epochs,
)

clf.initialize()

clf.module.load_state_dict(torch.load("model.pth"))

# com port connection
COMport = "COM7"
ser = serial.Serial(COMport, 38400, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)

# look for stream
print("Looking for streams")
streams = resolve_stream("name", "EventIDE_Signal_Stream")

# create inlet
print("test")
inlet = StreamInlet(streams[0])
print("Connected to 'EventIDE_Signal_Stream' stream")

# Read out stream
while True:
    print("test2`3")
    time.sleep(1)
    sample, timestamp = inlet.pull_chunk()
    if len(sample) == 0:
        continue  # skip if chunk contains no samples
        # timestamp.append(0)
        # sample.append(0)
    print("test3")
    print(f"Received sample at time {timestamp[0]}:")

    # # classify
    # sample = np.array(sample)
    # sample = sample.reshape(1, 19, 2501)
    # sample = torch.tensor(sample)
    # sample = sample.float()
    # sample = sample.to(device)
    # sample = clf.predict(sample)

    # [print(subsample) for subsample in sample]
    # bytesample=bytearray(sample)
    comsample = f"S{sample[0]}E"
    print(sample[0])
    ser.write(bytes(comsample, "utf-8"))
