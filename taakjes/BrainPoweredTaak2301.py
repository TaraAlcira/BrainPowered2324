from psychopy import visual, core, event
import serial
import numpy as np
import time
import random
from psychopy.hardware import keyboard
import sys

#function for running the experiment
def run_task(epochs, features, pause, trail_length):
    '''
    Help me
    '''
    
    for epoch in range(epochs):
        # Create a new dictionary with shuffled key-value pairs
        keys = list(features.keys())
        random.shuffle(keys)
        random_feats = {key: features[key] for key in keys}
        print(random_feats)    
        # determine after how many trials you want a break
        if (epoch%pause == 0 and epoch!=0):    
            message = visual.TextStim(win, text='Pauze!\nDruk op spatie om verder te gaan')
            message.draw()
            win.flip()
            event.waitKeys() 
        
        for feature in random_feats.keys():
            #feature instruction
            message = visual.TextStim(win, text=feature)
            message.draw()
            win.flip()
            
            # Show feature instruction for 3 seconds
            for i in range(3):
                text = feature + '\n' + str(3-i)
                message = visual.TextStim(win, text=text)
                message.draw()
                win.flip()
                core.wait(1)

            
            #fixation cross to indicate start EEG measurement
            message = visual.TextStim(win, text='x')
            message.draw()
            win.flip()
            
            # Uncomment code below to send markers to EEG
            specific_marker = random_feats[feature]
            markerValue = specific_marker.to_bytes(1, byteorder="big")
            serPort.write(markerValue)
            specific_marker += 1

            # Time to show the stimulus
            core.wait(trail_length)
            
            #stop program if escape is pressed
            if event.getKeys(keyList=["escape"]):
                core.quit()



#Settings of experiment
serPort = serial.Serial("COM6", 19200, timeout=1) # Use device manager on your stimulus laptiop to get the right COM
win = visual.Window(fullscr=True) #window fulscreen

#features are in dictionary, so that the number can idicate a corresponding marker
features = {"Beweeg rechter hand" : 1, "Beweeg linker hand" : 2, "Beweeg beide handen" :3, "Doe niks" :4} #features that will be measured

#first window: waiting for any keypress to start experiment
message = visual.TextStim(win, text='Druk op spatie om de EEG meting te starten')
message.draw()
win.flip()
event.waitKeys()

# Run the task
run_task(90, features, 8, 5)

# stop experiment 
message = visual.TextStim(win, text='Het taakje is nu klaar!\nBedankt voor het meten! :)')
message.draw()
win.flip()
event.waitKeys()

# End the task
win.close()

##second part of the experiment. To measure a feature per group. 
#
##Group 1
#features_1 = {"Right Imagery" : 1, "Left Imagery" : 2, "Left Movement" :3, "Right Movement" :4} #features that will be measured
##Group 2
#features_2 = {"Right Imagery" : 1, "Left Imagery" : 2, "Left Movement" :3, "Right Movement" :4} #features that will be measured
##Group 3
#features_3 = {"Right Imagery" : 1, "Left Imagery" : 2, "Left Movement" :3, "Right Movement" :4} #features that will be measured
#
##start part 2 of the experiment
#message = visual.TextStim(win, text='Dit is het 2de deel van het experiment. Hier zal ...., .... en ... worden gemeten. \nKlik op een toets om verder te gaan')
#message.draw()
#win.flip()
#event.waitKeys() 
#
##run experiment 2.1
#experiment_runner(trials, features_1, 15)
#message = visual.TextStim(win, text='Deel 2.1 is nu klaar. Sla het EEG document op onder feature_1_Naam_dag_maand. Start een nieuw EEG document.\nKlik op een toets om verder te gaan naar feature_2')
#message.draw()
#win.flip()
#event.waitKeys() 
#
##run experiment 2.2
#experiment_runner(trials, features_2, 15)
#message = visual.TextStim(win, text='Deel 2.2 is nu klaar. Sla het EEG document op onder feature_2_Naam_dag_maand. Start een nieuw EEG document.\nKlik op een toets om verder te gaan naar feature_2')
#message.draw()
#win.flip()
#event.waitKeys() 
#
##run experiment 2.3
#experiment_runner(trials, features, 15)
#message = visual.TextStim(win, text='Alle features zijn gemeten! klik op een toets om het experiment te sluiten. Bedankt voor het meedoen :)')
#message.draw()
#win.flip()
#event.waitKeys() 
#
#
