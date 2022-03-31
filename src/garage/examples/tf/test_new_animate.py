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

pids = pids[0:20]

window_length = 800
window_height = 800 


scale_z = window_length/(267+267)
scale_r = window_height/27

batch = pyglet.graphics.Batch()



class Tracks: 
    def __init__(self, event, pids, rlist, zlist): 
        self.pid = pids
        
        event['z'] = event['z'].values * scale_z + 266  
        event['r'] = event['r'].values * scale_r

        self.i = 0 
        self.last_add = 0 
        self.circles = [] 
        self.rlist = rlist.values* scale_r 
        self.zlist = zlist.values * scale_z + 266 

    def add_point(self, pid, dt): 
        self.i =0 
        self.data = event[event['particle_id'] == self.pid] 
        #for i in range(len(self.data)):
        while (self.i < len(self.data)) & (time.time() - self.last_add > dt): 
            print(time.time() - self.last_add > dt, self.i)
            hit = self.data.iloc[self.i, :]
            self.circles.append(shapes.Circle(hit.z, hit.r, 5, color=(60,60,60), batch=batch))
            self.last_add = time.time() 
            self.i += 1 

    def track_length(self):
        return len(self.data) 

    def is_finished(self):
        if self.i >  self.track_length()-1: 
            finished = True
        else: 
            finished = False
        return finished 

    def add_point_rf(self, dt, pid): 
        self.i = 0  
        indices = np.where(pids.values.flatten() == pid)[0]
        for i in range(len(indices)): 
            if time.time() - self.last_add > dt: 
                self.circles.append(shapes.Circle(self.zlist[self.i], self.rlist[self.i], 5, color=(250,0,0), batch=batch))
                self.last_add = time.time() 
                self.i += 1 

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
        for particle in np.unique(pids): 
            self.pid = particle
            self.add_point(10, particle)
            self.add_point_rf(0.5, particle)
            
            label = pyglet.text.Label("Particle id: "+str(particle), font_size=20, x=0, y=0, color=(255, 0, 0, 0), batch=batch) 
            label.color = (0, 0, 100, 255)
            if self.is_finished_rf(): 
                circles = []
                print("finishhhed")
                label.delete() 




window = pyglet.window.Window(window_length+100, window_height+100)
#pyglet.sprite.Sprite(a['head'][0], x=self.cor_x, y=self.cor_y, batch=batch, group=foreground)

glClearColor(255, 255, 255, 1.0) # red, green, blue, and alpha(transparency)


@window.event
def on_draw():
    window.clear()
    batch.draw()

def dummy(dt): 
    #print("dummy called") 
    dummy = 0 


def visualise(event, pids, rlist, zlist): 
    tracks = Tracks(event, pids, rlist, zlist)
    tracks.plot_tracks() 
    pyglet.clock.schedule_interval(dummy, 0.5)
    pyglet.app.run()
    


visualise(event, pids, r, z)
