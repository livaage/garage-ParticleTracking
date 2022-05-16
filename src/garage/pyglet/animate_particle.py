import pyglet 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pyglet import shapes
from pyglet.gl import glClearColor
import time 



event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
r = pd.read_csv('~/garage/src/garage/examples/tf/g_r.csv', header=None)
z = pd.read_csv('~/garage/src/garage/examples/tf/g_z.csv', header=None)
pids = pd.read_csv('~/garage/src/garage/examples/tf/g_pids.csv', header=None)


#todo: 
# Draw detector lines 


circles = [] 
batch = pyglet.graphics.Batch()

i = 0 



window_length = 800
window_height = 800 


scale_z = window_length/(267+267)
scale_r = window_height/27

event['z'] = event['z'].values * scale_z + 266  
event['r'] = event['r'].values * scale_r


window = pyglet.window.Window(window_length+100, window_height+100)
#pyglet.sprite.Sprite(a['head'][0], x=self.cor_x, y=self.cor_y, batch=batch, group=foreground)

class Track:
    def __init__(self, pid): 
        self.pid = pid
        self.data = event[event['particle_id'] == self.pid] 
        self.i = 0 
        self.last_add = 0 

    def add_point(self, dt): 
        if self.i > (self.track_length() -1): 
            #time_ended = time.time() 
            pass
        else: 
            if time.time() - self.last_add > dt: 
                hit = self.data.iloc[self.i, :]
                circles.append(shapes.Circle(hit.z, hit.r, 5, color=(60,60,60), batch=batch))
                self.last_add = time.time() 
                self.i +=1 

    def track_length(self):
        return len(self.data) 

    def is_finished(self):
        if self.i >  self.track_length()-1: 
            finished = True
        else: 
            finished = False
        return finished 

class rfTrack:
    def __init__(self, rlist, zlist, pid): 
        self.i = 0 
        self.last_add = 0 
        indices = np.where(pids == pid)
        self.rlist = rlist.values[indices] * scale_r 
        self.zlist = zlist.values[indices] * scale_z + 266 

    def add_point(self, dt): 
        if self.i > self.track_length() -1: 
            pass
        else:
            if time.time() - self.last_add > dt: 
                circles.append(shapes.Circle(self.zlist[self.i], self.rlist[self.i], 5, color=(250,0,0), batch=batch))
                self.last_add = time.time() 
                self.i +=1 

    def track_length(self):
        return len(self.rlist) 

    def is_finished(self):
        if self.i > self.track_length() - 1: 
            finished = True
#            window.close() 
        else: 
            finished = False
        return finished 

glClearColor(255, 255, 255, 1.0) # red, green, blue, and alpha(transparency)



track1 = Track(-17737)
track2 = rfTrack(r, z, -17737)

label = pyglet.text.Label("Particle id:  -17737", font_size=20, x=0, y=0, color=(255, 0, 0, 0)) 
label.color = (0, 0, 100, 255)


def dummy(dt): 
    #print("dummy called") 
    dummy = 0 

@window.event
def on_draw():
    window.clear()
    label.draw()
    track1.add_point(0.5)
#    if track1.is_finished():
#        track2.add_point(0.05)
#    closing_time()

    batch.draw() 

#@window.event
#def closing_time(): 
#    if track2.is_finished() ==True: 
#        pyglet.app.exit()

def visualise(): 
 
    #breaks without the dummy call, no idea why 
    pyglet.clock.schedule_interval(dummy, 0.5)

    if __name__ == '__main__':

        pyglet.app.run()

