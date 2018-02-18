from bokeh.io import output_file, show
from bokeh.layouts import layout, column, row
from bokeh.models.widgets import Panel, Tabs, Slider
from bokeh.plotting import curdoc
from bokeh.models import ColumnDataSource
from images_class import *

from bokeh.client import push_session
from bokeh.driving import cosine

#The dashboard .py file serves to call the different image transformations out of the images_class
#The output file is scratch.html when no server is started. See last line for further explenation.
output_file("dashboard.html", title="Processed Photo - Dashboard")

#Open the image "image.jpg" as an original image, do not transform it and assign its figure to p1_img for later plotting
p1 = Original_Image("photo.jpeg")
p1_img = p1.create_img()

#Open the image "image.jpg", seperate its color Value "RED" and assign its figure to p2_img for later plotting
p2 = Colorchannel_Image("photo.jpeg","R")
p2.transform_img()
p2_img = p2.create_img()

#Open the image "image.jpg", seperate its color Value "GREEN" and assign its figure to p3_img for later plotting
p3 = Colorchannel_Image("photo.jpeg","G")
p3.transform_img()
p3_img = p3.create_img()

#Open the image "image.jpg", seperate its color Value "BLUE" and assign its figure to p4_img for later plotting
p4 = Colorchannel_Image("photo.jpeg","B")
p4.transform_img()
p4_img = p4.create_img()

#Open the image "image.jpg", convert it to greyscale in the images_class and assign its figure to p5_t1_img for later plotting, assign it to a tab
p5_t1 = Greyscale_Image("photo.jpeg", [0.3, 0.59, .11])
p5_t1.transform_img()
p5_t1_img = p5_t1.create_img()
tab1 = Panel(child=p5_t1_img, title="Greyscale")

#Open the image "image.jpg", calculate the color cubes, redefine each color to its nearest color-cube value and assign its figure to p5_t2_img for later plotting, assign it to a tab
bits = 16
p5_t2 = Mediancut_Image("photo.jpeg", bits, threading=False)
p5_t2.transform_img()
p5_t2_img = p5_t2.create_img()
tab2 = Panel(child=p5_t2_img, title="{}-Bit".format(str(bits)))

#Open the image "image.jpg", add random noise to it and assign its figure to p6_img. Add a slider to it to let the user input values
p6 = Noise_Image("photo.jpeg")
image = p6.transform_img(.1)
source = ColumnDataSource({'image': [image]})
p6_img = figure(title="Photo, Noise", x_range=(0, len(image[0])), y_range=(0, len(image)), plot_width=len(image[0]) // size, plot_height=len(image) // size, toolbar_location=None)
p6_img.image_rgba(image='image', x=0, y=0, dw=len(image[0]), dh=len(image), source=source)
slider = Slider(start=0, end=.5, value=.1, step=.01, title="Noise Level")

#Open the image "image.jpg", apply a gaussian filter to it and assign its figure to p5_t2_img for later plotting, assign it to a tab
p7 = Gaussian_Image("photo.jpeg",2, threading=False)
p7.transform_img()
p7_img = p7.create_img()

#Function that recalculates the noise in an image on slider change
def update_img(attr, old, new):
    new_img = p6.transform_img(new)
    source.data = {'image': [new_img]}
slider.on_change("value", update_img)

#Combine the tabs inside a Tag-Object
tabs = Tabs(tabs=[tab1, tab2])

#START BOKEH SERVER FIRST IN SAME DIRECTORY: CMD: --> CD "THIS-DIRECTORY" --> BOKEH SERVE
session = push_session(curdoc())
session.show(layout([row(column(p1_img, row(p2_img, p3_img, p4_img)), column(row(tabs), p6_img, slider, p7_img))]))
session.loop_until_closed()
#To make the program work without the bokeh server, uncomment the line underneath and comment out the three lines above
#show(layout([row(column(p1_img, row(p2_img, p3_img, p4_img)), column(row(tabs), p6_img, slider, p7_img))]))
