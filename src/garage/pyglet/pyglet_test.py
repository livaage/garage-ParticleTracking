import pyglet 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pyglet import shapes
from pyglet.gl import glClearColor




#todo: 
# Make sure it's animated
# Draw detector lines 
# Integrate with rf
# Commadn line animation 

#line1 = shapes.Line(co_x1, co_y1, co_x2, co_y2, width, color = (50, 225, 30), batch = batch)



event = pd.read_hdf('~/gnnfiles/data/ntuple_PU200_numEvent1000/ntuple_PU200_event0.h5')
r = pd.read_csv('~/garage/src/garage/examples/tf/g_r.csv', header=None)
z = pd.read_csv('~/garage/src/garage/examples/tf/g_z.csv', header=None)
pids = pd.read_csv('~/garage/src/garage/examples/tf/g_pids.csv', header=None)
gevent = pd.DataFrame({'particle_id': pids.values.flatten(), 
                      'z': z.values.flatten(),
                      'r': r.values.flatten()})


e_particle = event[event['particle_id']==-17737] 
rf_particle = gevent[gevent['particle_id']==-17737] 

batch = pyglet.graphics.Batch()
window_length = 1000
window_height = 1000 

scale_z = 1000/(267+267) 
scale_r = 1000/27 


e_particle['z'] = e_particle['z'].values*scale_z + 266
e_particle['r'] = e_particle['r'].values*scale_r 
rf_particle['z'] = rf_particle['z'].values*scale_z + 266
rf_particle['r'] = rf_particle['r'].values*scale_r


window = pyglet.window.Window(window_length, window_height)
#pyglet.sprite.Sprite(a['head'][0], x=self.cor_x, y=self.cor_y, batch=batch, group=foreground)


glClearColor(255, 255, 255, 1.0) # red, green, blue, and alpha(transparency)

#circle = shapes.Circle(-41,  3.11, 300, color=(50,225,30), batch=batch) 


label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')

#circle = shapes.Circle(700, 150, 100, color=(50, 225, 30), batch=batch)


circles = [] 
def add_point(index, is_MC): 
    if is_MC: 
        hit = e_particle.iloc[index, ]
        color1 = 60
    else: 
        hit = rf_particle.iloc[index, ] 
        color1 = 160 

    circles.append(shapes.Circle(hit.z, hit.r, 15, color=(color1,60,60), batch=batch)) 


def update(dt): 
    for i in range(10): 
        add_point(i, 1) 
        add_point(i, 0)

@window.event
def on_draw():
    window.clear()
    #label.draw()
    batch.draw() 

pyglet.clock.schedule_interval(update, 3)
if __name__ == '__main__':
#    for i in np.arange(0,10000, radius):
#        for j in np.arange(0,10000, radius):
#            itcount = 0
#            if i==180*5 and j ==180*5 and itcount<1: 
#                state=1
#                itcount = 1
#                intbpstate = 'head'
#            else: 
#                state=0
#                intbpstate = 'nostate'
#            cell_dict[(i,j)].draws(intbpstate)
#            batch.draw()
#  for i in range(10): 
 #     add_point(i, 1) 
     x = 10
     # batch.draw() 


pyglet.app.run()

