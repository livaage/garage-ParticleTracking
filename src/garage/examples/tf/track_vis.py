from tkinter import N
from turtle import window_width
import pyglet 
import pandas as pd 
import numpy as np 
from pyglet import shapes
from pyglet.gl import glClearColor
from pyglet import clock

rf_file = pd.read_csv('/home/lhv14/garage/src/garage/examples/tf/garage_outputs.csv')

batch = pyglet.graphics.Batch()
window_length = 1000
window_height = 1000 

scale_z = window_length/(2*267) 
scale_r = window_height/27 

n_track_hits = 7

rf_file['mc_z'] = rf_file['mc_z'].values*scale_z + 266
rf_file['mc_r'] = rf_file['mc_r'].values*scale_r 
rf_file['pred_z'] =rf_file['pred_z'].values*scale_z + 266
rf_file['pred_r'] = rf_file['pred_r'].values*scale_r


p1 = rf_file[rf_file['particle_id']==-18951]

# sample every 100th particle id 
pids = rf_file.particle_id.values[::100]
files = rf_file.filenumber.values[::100]

window = pyglet.window.Window(window_length, window_height)


label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')


class Point:
    def __init__(self): 
        self.circles = [] 
        self.i = 0 
        self.pid_counter = 0 
        self.pid = pids[0]
        self.filenumber = files[0]
    
    def plot_point(self, dt): 
        self.particle = rf_file[(rf_file['particle_id']==self.pid) & (rf_file['filenumber']==self.filenumber)]

        
        color1 = 160
        #print("i is now ", self.i)
        if self.i < (n_track_hits-1): 
         hit = self.particle.iloc[self.i, ]   
         self.circles.append(shapes.Circle(hit.mc_z, hit.mc_r, 5, color=(color1,60,60), batch=batch)) 
         self.circles.append(pyglet.text.Label("Particle id:  " + str(self.pid) + "  After training on " + str(self.pid_counter*10) +"tracks", font_size=12, batch=batch))

         self.i += 1 

        elif (self.i > (n_track_hits-2)) & (self.i < (n_track_hits*2-2)): 
            hit = self.particle.iloc[self.i-(n_track_hits-1), ]
            color3 = 2014
            self.circles.append(shapes.Circle(hit.pred_z, hit.pred_r, 5, color=(0,60,color3), batch=batch)) 
            self.i+=1 

        else: 
            self.i = 0 
            self.pid_counter += 1 
            self.pid = pids[self.pid_counter]
            self.filenumber = files[self.pid_counter]
            #del(self.circles)
            self.circles = []
            #self.particle = rf_file[rf_file['particle_id']==self.pid]



p = Point() 

clock.schedule_interval(p.plot_point, 0.1)
frame = 0 
@window.event
def on_draw():
    global frame 
    window.clear()
    frame += 1 
    batch.draw() 
    pyglet.image.get_buffer_manager().get_color_buffer().save('screenshots/screenshot'+str(frame)+'.png')    #label.draw()
    #image_count += 1 


pyglet.app.run()
