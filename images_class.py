from bokeh.plotting import figure
import numpy as np
from PIL import Image
import random
import threading

#Defined size, every little image_plot in the dashboard is factor "size" smaller than the big one.
size = 3

#Base Image class, opens up an image, defines its ranges and assigns the image to a np array. Offers different featuers like downscaling an image
class Plot_Image():
    def __init__(self, img):
        rgb_image = Image.open(img).convert("RGBA")
        self._xdim, self._ydim = rgb_image.size
        self._image = np.flipud(np.array(rgb_image))

    #Takes an image and creates a new one while only taking every factor-th pixel, so reducing its dimensions by the factor
    def downscale(self, factor):
        print("Downscaling Image by factor of {}...".format(factor))
        #Creating a new empty image
        new_img = []
        #the new image shall have factor-less pixels in each dimension, so only iterate width//factor and height//factor times
        for column in range(len(self._image)//factor):
            #Create new row in the new_image
            new_img.append([])
            for pixel in range(len(self._image[column])//factor):
                #Take the pixel in the current*factor row and current*factor column, assign it to the new image
                new_img[column].append(list(self._image[column*factor][pixel*factor]))
        #overwrite the class image
        self._image = np.array(new_img)
        #update dimensions BUT multiply by factor, to make uniformal scaling all plots via "size" possible
        self._ydim = len(self._image)*factor
        self._xdim = len(self._image[0])*factor

#Class to just plot an image without transforming it
class Original_Image(Plot_Image):
    #constructor is inherited, only need to define create_image class which returns a bokeh figure to allow plotting
    def create_img(self):
        image_figure = figure(title="Original Image", x_range=(0, self._xdim), y_range=(0, self._ydim), plot_width=self._xdim,
                              plot_height=self._ydim)
        image_figure.image_rgba(image=[self._image], x=0, y=0, dw=self._xdim, dh=self._ydim)
        return image_figure

#Class to isolate a color channel of an image
class Colorchannel_Image(Plot_Image):
    #The additional attribute "channel" is delivered, which tells us what channel to isolate, Either R,G,B as STRING
    def __init__(self, img, channel):
        Plot_Image.__init__(self, img)
        self._channel = channel.upper()

    #Seperate the color channel
    def transform_img(self):
        #Downscale image first, cause its only presented quiet small on the dashboard, so we don't need all this pixels and it makes the process faster
        self.downscale(4)
        print("Separating {}-Color Channel...".format(self._channel))
        channel_dict = {"R":0, "G":1, "B":2}
        #For every pixel in the image, null all channels except the one that is to seperate
        for column in self._image:
            for pixel in column:
                for color_channel in range(3):
                    if color_channel == channel_dict[self._channel]:
                        continue
                    pixel[color_channel] = 0

    #Return the image as a bokeh figure, make the plot smaller by the factor "size"
    def create_img(self):
        image_figure = figure(title="Image, only {}-Channel".format(self._channel), x_range=(0, self._xdim), y_range=(0, self._ydim), plot_width=self._xdim//size,
                              plot_height=self._ydim//size, toolbar_location=None)
        image_figure.image_rgba(image=[self._image], x=0, y=0, dw=self._xdim, dh=self._ydim)
        return image_figure

#Class that takes an image and transforms it to a greyscale with a weighted value for each color channel
class Greyscale_Image(Plot_Image):
    def __init__(self, img, weight):
        Plot_Image.__init__(self, img)
        self._weight = weight

    def transform_img(self):
        #Downscale again for the same reasons
        self.downscale(4)
        print("Converting Image to Greyscale...")
        z = 0
        #For every pixel in the image, sum up all its weighted pixels, devide value by 3 and assign this greyscale value to all color channels
        for column in self._image:
            for pixel in column:
                grey_value = sum([x*y for x, y in zip(pixel[:3], self._weight)])
                for color_channel in range(3):
                    pixel[color_channel] = grey_value
            z += 1

    # Return the image as a bokeh figure, make the plot smaller by the factor "size"
    def create_img(self):
        image_figure = figure(title="Greyscale Image", x_range=(0, self._xdim), y_range=(0, self._ydim), plot_width=self._xdim//size,
                              plot_height=self._ydim//size, toolbar_location=None)
        image_figure.image_rgba(image=[self._image], x=0, y=0, dw=self._xdim, dh=self._ydim)
        return image_figure

#Class that takes an image, reduces its color via median-cut algorithm and returns a image with n-bits. Threading supported
class Mediancut_Image(Plot_Image):
    def __init__(self, img, n, threading=False):
        Plot_Image.__init__(self, img)
        self._n = n
        self._threading = threading

    #Here is where the magic happens to calculate the medians and the new pixel values. More detailed comments inline
    def transform_img(self):
        #Again, downscale the image to save time.
        self.downscale(4)
        print("Reducing Colors to {}-Bit... (May take a few seconds, please do not Interrupt)".format(self._n))
        #First, we create a color-cube that is just a list containing ALL pixels of the image after another. So we have a color-cube with 0 subcubes yet.
        pixels = []
        for column in self._image:
            for pixel in column:
                pixels.append(pixel)

        color_cube = [pixels]

        color_split = 0
        #Now, we perform median cut as long as we have less than n subcubes, while n is the number of colors the user wanted the end-image to have
        while len(color_cube) < self._n:
            #We don't know which cube we have to cut and for which color, so we set impossible values for the start to force at least one cube to overwrite it
            max_difference = -1
            color_split = -1
            cube = -1

            #Next, for every subcube in the image, find the maximal difference in each color channel
            for subcube in range(len(color_cube)):
                subcube_red_difference = max([pixel[0] for pixel in color_cube[subcube]]) - min([pixel[0] for pixel in color_cube[subcube]])
                subcube_green_difference = max([pixel[1] for pixel in color_cube[subcube]]) - min([pixel[1] for pixel in color_cube[subcube]])
                subcube_blue_difference = max([pixel[2] for pixel in color_cube[subcube]]) - min([pixel[2] for pixel in color_cube[subcube]])

                #If the new found maximal difference value is  higher than the value found previously, we got a new match which colour to separate in which subcube
                if max_difference < max([subcube_red_difference, subcube_green_difference, subcube_blue_difference]):
                    cube = subcube
                    max_difference = max([subcube_red_difference, subcube_green_difference, subcube_blue_difference])
                    if subcube_red_difference == max_difference:
                        color_split = 0
                    elif subcube_green_difference == max_difference:
                        color_split = 1
                    elif subcube_blue_difference == max_difference:
                        color_split = 2
                subcube += 1

            #Now, we define two clusters, one for each side of the subcube
            cluster_1 = []
            cluster_2 = []
            #Calculate the median color value of the color which is to split via numpy
            current_median = np.median([pixel[color_split] for pixel in color_cube[cube]])
            #Now, assign each pixel in the color-cube to one of the two clusters
            for pixel in color_cube[cube]:
                if pixel[color_split] < current_median:
                    cluster_1.append(pixel)
                    continue
                cluster_2.append(pixel)
            #Finally, we overwrite the original color-subcube by one of our cluster, and append the other cluster to the end of the color-cubes, therefore making it a new subcube
            color_cube[cube] = cluster_1
            color_cube.append(cluster_2)

        #Now that we have defined n-subcubes in our colorcube, we need to assign each pixel in our original image to one of these subcubes.
        mean_values = []
        #First of all(), we need to calculate the mean values of each subcube again, which will then hold as the reference color of this subcube.
        #We define a new list containing the median values of the subcubes, the first entry of the list is the median of the first color subcube etc.
        for cluster in range(len(color_cube)):
            median_red = np.mean([pixel[0] for pixel in color_cube[cluster]])
            median_green = np.mean([pixel[1] for pixel in color_cube[cluster]])
            median_blue = np.mean([pixel[2] for pixel in color_cube[cluster]])
            mean_values.append([int(median_red), int(median_green), int(median_blue)])
            cluster += 1

        #This lcoal function loops through all pixels in the column, finds its nearest subcube median and assign its color values to itself
        def s(column, z):
            column = self._image[column]
            #For every pixel in the column, find the closest matching median color of a subcube
            for pixel in column:
                nearest = -1
                nearest_distance = 10000
                #Calculate the euclidian distance to each median color of each subcube, find the shortest one
                for mean in range(len(mean_values)):
                    distance = np.linalg.norm(mean_values[mean] - pixel[:3])
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest = mean
                    mean += 1
                #assign the color of the nearest median-color-cube to the pixel
                for i in range(3):
                    pixel[i] = int(mean_values[nearest][i])

        #loop through the columns of the image and find the new pixel values, we do this seperately to allow the user to thread this process.
        #If he wants to, every row is processed by a new thread. I found out that this sometimes gives strange results, but I'm not sure why.
        #So I made it optional to the user, when he does not want to thread, the columns are processed after another.
        z = 0
        for column in range(len(self._image)):
            if self._threading:
                threading._start_new_thread(s, (column, z))
            else:
                s(column, z)
            z += 1

    #Finally, create an image figure and return it for later plotting in the dashboard
    def create_img(self):
        image_figure = figure(title="Image, {}-Bit".format(str(self._n)), x_range=(0, self._xdim), y_range=(0, self._ydim),
                              plot_width=self._xdim // size,
                              plot_height=self._ydim // size, toolbar_location=None)
        image_figure.image_rgba(image=[self._image], x=0, y=0, dw=self._xdim, dh=self._ydim)
        return image_figure

#The next class is to noise an image with random Salt-Pepper noise
class Noise_Image(Plot_Image):
    def __init__(self, img):
        Plot_Image.__init__(self, img)

    #Transformation needs a noise level between 0 and 1, where 1 = 100& of the image pixels are covered in noise.
    def transform_img(self, noise=.10):
        print("Noising with noice level {}...".format(noise))
        #make a copy of the image to be able to later reduce noise again
        img = self._image.copy()
        #loop for as many times as there is noise to be inserted, to with 100 pixels and noise levl .1, we need to loop 5 times, each time making a random pixel white and another one black
        for i in range(int(self._xdim*self._ydim*noise/2)):
            #Making a random pixel white, another random pixel black. I know, theres a chance that I hit one pixel twice, but the process is much faster this way than iterating over all pixels
            img[random.randint(0,self._ydim-1)][random.randint(0,self._xdim-1)] = [0,0,0,255]
            img[random.randint(0,self._ydim-1)][random.randint(0,self._xdim-1)] = [255,255,255,255]
        return img

    #as always, return image figure for bokeh plotting in dashboard
    def create_img(self):
        image_figure = figure(title="Image, Noise", x_range=(0, self._xdim), y_range=(0, self._ydim), plot_width=self._xdim//size,
                              plot_height=self._ydim//size, toolbar_location=None)
        image_figure.min_border_top = 300
        return image_figure

#The last class is to apply a gaussian filter over an image, again with optional threading
class Gaussian_Image(Plot_Image):
    def __init__(self, img, sigma, threading=False):
        Plot_Image.__init__(self, img)
        self._sigma = sigma
        self._xdim = self._xdim * 4
        self._ydim = self._ydim * 4
        self._threading = threading

    def transform_img(self):
        #again, downscale the image to reduce processing costs
        self.downscale(4)
        print("Applying Gaussian Filter... (May take a few seconds, please do not Interrupt)")
        #this function applies the filter on each row of the image
        def t(row):
            #calculate the dimensions of the filter in each axis direction <--dim-- O --dim-->
            filter_dimension = int(len(gfilter) // 2)
            #for each pixel in the row
            for column in range(filter_dimension, len(self._image[row]) - filter_dimension):
                #for each color in the pixel, leaving out a gap on the right and left where the filter would overflow
                for color in range(3):
                    new_value = 0
                    #now, go through the filter and, for every filter element, add up the new value by the corresponding image pixel multiplied by the filter value
                    for i in range(-filter_dimension, filter_dimension + 1):
                        for j in range(-filter_dimension, filter_dimension + 1):
                            new_value += self._image[row + i][column + j][color] * gfilter[i + filter_dimension][
                                j + filter_dimension]
                    #assign the pixel color the new value constructed with the filter
                    working_img[row][column][color] = new_value

        #1D-Representation of the filter, manually entered. I'm not sure how I would calcullate that :/
        gfilter = [2,4,5,4,2,4,9,12,9,4,5,12,15,12,5,4,9,12,9,4,2,4,5,4,2]
        s = sum(gfilter)
        #divide the filter by the sum of its elements cause we would get too bright pixels otherwise, the filter needs to sum up to 1
        gfilter = [x / s for x in gfilter]
        #reshape the filter to make it 2D
        gfilter = np.array(gfilter).reshape((5,5))
        #length in each direction is length of the filter // 2
        filter_dimension = int(len(gfilter)//2)
        working_img = self._image[:]
        #For each row in the image, call the blurr function, either with threading or not.
        #Again, threading sometimes gives strnage results, so at default it is turned off.
        for t_row in range(filter_dimension,len(self._image)-filter_dimension):
            if self._threading:
                threading._start_new_thread(t, (t_row,))
            else:
                t(t_row)
        self._image = working_img

    #Finally, return an image figure
    def create_img(self):
        image_figure = figure(title="Image, Gaussian Blur", x_range=(0, self._xdim), y_range=(0, self._ydim), plot_width=self._xdim//size,
                              plot_height=self._ydim//size, toolbar_location=None)
        image_figure.image_rgba(image=[self._image], x=0, y=0, dw=self._xdim, dh=self._ydim)
        return image_figure