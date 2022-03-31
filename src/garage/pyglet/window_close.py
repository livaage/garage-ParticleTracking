# importing pyglet module
import pyglet
import pyglet.window.key
 
# width of window
width = 500
 
# height of window
height = 500
 
# caption i.e title of the window
title = "Geeksforgeeks"
 
# creating a window
window = pyglet.window.Window(width, height, title)
 
# text
text = "GeeksforGeeks"
 
# creating a label with font = times roman
# font size = 36
# aligning it to the center
label = pyglet.text.Label(text,
                          font_name ='Times New Roman',
                          font_size = 36,
                          x = window.width//2, y = window.height//2,
                          anchor_x ='center', anchor_y ='center')
 
# on draw event
@window.event
def on_draw():
     
    # clearing the window
    window.clear()
     
    # drawing the label on the window
    label.draw()
    #window.close() 
     
# key press event   
@window.event
def on_key_press(symbol, modifier):
     
    # key "E" get press
    if symbol == pyglet.window.key.E:
         
        # close the window
        window.close()
 
 
# start running the application
pyglet.app.run()
