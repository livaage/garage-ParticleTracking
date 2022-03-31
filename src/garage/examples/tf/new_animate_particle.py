import pyglet 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pyglet import shapes
from pyglet.gl import glClearColor
import time 


#event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
#r = pd.read_csv('~/garage/src/garage/examples/tf/g_r.csv', header=None)
#z = pd.read_csv('~/garage/src/garage/examples/tf/g_z.csv', header=None)
#pids = pd.read_csv('~/garage/src/garage/examples/tf/g_pids.csv', header=None)

window_length = 800
window_height = 800 


scale_z = window_length/(267+267)
scale_r = window_height/27

batch = pyglet.graphics.Batch()

window = pyglet.window.Window(window_length+100, window_height+100)
#pyglet.sprite.Sprite(a['head'][0], x=self.cor_x, y=self.cor_y, batch=batch, group=foreground)

glClearColor(255, 255, 255, 1.0) # red, green, blue, and alpha(transparency)



class Tracks: 
    def __init__(self, event, pids, rlist, zlist): 
        self.pid = pids
        
        event['z'] = event['z'].values * scale_z + 266  
        event['r'] = event['r'].values * scale_r

        self.data = event[event['particle_id'] == self.pid] 
        self.i = 0 
        self.last_add = 0 
        self.circles = [] 
        self.rlist = rlist.values* scale_r 
        self.zlist = zlist.values * scale_z + 266 

    def add_point(self, dt, pid): 
        for i in len(self.data):
            if time.time() - self.last_add > dt: 
                hit = self.data.iloc[self.i, :]
                self.circles.append(shapes.Circle(hit.z, hit.r, 5, color=(60,60,60), batch=batch))
                self.last_add = time.time() 
          

    def track_length(self):
        return len(self.data) 

    def is_finished(self):
        if self.i >  self.track_length()-1: 
            finished = True
        else: 
            finished = False
        return finished 

    def add_point_rf(self, dt, pid): 
        
        indices = np.where(self.pids.values.flatten() == pid)
        for i in indices: 
            if time.time() - self.last_add > dt: 
                self.circles.append(shapes.Circle(self.zlist[self.i], self.rlist[self.i], 5, color=(250,0,0), batch=batch))
                self.last_add = time.time() 

    def track_length_rf(self):
        return len(self.rlist) 

    def is_finished_rf(self):
        if self.i > self.track_length() - 1: 
            finished = True
#            window.close() 
        else: 
            finished = False
        return finished 

    def plot_tracks(self): 
        for particle in np.unique(self.pids): 
            self.add_point(0.5, particle)
            self.add_point_rf(0.5, particle)
            
            label = pyglet.text.Label("Particle id: "+str(particle), font_size=20, x=0, y=0, color=(255, 0, 0, 0), bacth=batch) 
            label.color = (0, 0, 100, 255)




@window.event
def on_draw():
    window.clear()


def dummy(dt): 
    #print("dummy called") 
    dummy = 0 


def visualise(event, pids, rlist, zlist): 
    tracks = Tracks(event, pids, rlist, zlist)
    tracks.plot_tracks() 
    pyglet.clock.schedule_interval(dummy, 0.1)
    pyglet.app.run()
