Important: The python script rund on bokeh server. Follow the next Steps to make it run:
1.) Make sure the following files are located in the same folder: dashboard.py, images_class.py, __init__.py and image.jpg
2.) Open the Shell and navigate to the folder where the dashboard.py are located.
3.) Run the following command: bokeh serve
4.) Wait for bokeh to run the server, it finished when the id is displayed on the screen
5.) Next, open another shell, navigate into the same folder as previous and call: python3 dashboard.py, or python.exe dashboard.py
6.) In your second Shell, you see how the images are processed. Wait for the process to finish, do not interrupt it. When the process finished, a new findow will
open in your browser automatically. 
7.) To test the Tabulator, click the "16-Bit" tab and the new, color reduced image will be displayed.
8.) To test the noise producer, tab on a new position on the slider. DO NOT SLIDE, otherwise python will try to calcullate all intermediate values. 
9.) When you finished, close the browser tab or end the bokeh server via command-C. 

Additionally: The dasboard.py only serves to call the functions located in the images_class.py. This is where all the magic happens.
I implemented some of the transoformations to support threading, but it sometimes gives strange results when using it. So, by default, its turned off, 
but feel happy to try it out. 

I've read online that the bokeh version we use in the course may be incompatible with some tornado version, but I had no issues. If your computer has,
it is recommended to downgrade tornado to an older, supported version. 
I downscaled the little image to safe processor power and decrease processing time, feal free to comment the downscaling out or set it to 1. 

I used bokeh, numpy, PIL, random and threading. As an IDE, I used PyCharm on Windows 10, tested out on Mac OSX. 

Photo: https://www.pexels.com/photo/adventure-alps-climb-clouds-267104/

Hire us: [Software Entwickler in Zürich](https://polygon-software.ch)!
