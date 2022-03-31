import pyglet

window = pyglet.window.Window()


def draw_label(div_x, div_y): 



    label = pyglet.text.Label('Hello, world',
                              font_name='Times New Roman',
                              font_size=36,
                              x=window.width//div_y, y=window.height//div_x,
                              anchor_x='center', anchor_y='center')


    @window.event
    def on_draw():
        window.clear()
        label.draw() 

    pyglet.app.run()

