import dominate
import ntpath
from dominate.tags import meta, h3, table, tr, td, p, a, img, br
import os
from subprocess import Popen, PIPE
from skimage.transform import resize
from pathlib import Path
import math

from common import utils
from common.image import save_image

class HTML():
    """This HTML class allows us to save images and write texts into a single HTML file.

     It consists of functions such as <add_header> (add a text header to the HTML file),
     <add_images> (add a row of images to the HTML file), and <save> (save the HTML to the disk).
     It is based on Python library 'dominate', a Python library for creating and manipulating HTML documents using a DOM API.
    """

    def __init__(self, web_dir, title, opt, refresh=0):
        """Initialize the HTML classes

        Parameters:
            web_dir (str) -- a directory that stores the webpage. HTML file will be created at <web_dir>/index.html; images will be saved at <web_dir/images/
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.opt = opt

        self.title = title
        self.web_dir = Path(web_dir)
        self.img_dir = self.web_dir.joinpath("images")
        self.img_dir.mkdir(parents=True, exist_ok=True)

        self.doc = dominate.document(title=title)
        
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir

    def add_header(self, text):
        """Insert a header to the HTML file
        Parameters:
            text (str) -- the header text
        """
        with self.doc:
            h3(text)

    def add_images(self, ims, txts, links, width=400):
        """add images to the HTML file
        Parameters:
            ims (str list)   -- a list of image paths
            txts (str list)  -- a list of image names shown on the website
            links (str list) --  a list of hyperref links; when you click an image, it will redirect you to a new page
        """

        ncol = 6
        nrow = math.ceil( len(ims) / ncol )
        
        for irow in range(nrow):

            itable = table(border=1, style="table-layout: fixed;")  # Insert a table
            self.doc.add(itable)
        
            with itable:
                with tr():                  
                    
                    # for im, txt, link in zip(ims, txts, links):
                    for icol in range(ncol):
                        idx = ncol * irow + icol 
                        if idx >= len(ims): break
                        
                        with td(style="word-wrap: break-word;", halign="center", valign="top"):
                            with p():
                                with a(href=links[idx]):
                                    img(style="width:%dpx" % width, src=ims[idx])
                                br()
                                p(txts[idx])


    def save(self):
        """save the current content to the HMTL file"""
        
        path_html = self.web_dir.joinpath("index.html")

        with open(str(path_html), 'wt') as f:
            f.write(self.doc.render())


    def save_images(self, visuals, image_path, aspect_ratio=1.0, width=256):
        """Save images to the disk.

        Parameters:
            webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
            visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
            image_path (str)         -- the string is used to create image paths
            aspect_ratio (float)     -- the aspect ratio of saved images
            width (int)              -- the images will be resized to width x width

        This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
        """
        image_dir = self.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        short_path1 = ntpath.basename(ntpath.dirname(image_path[0]))
        short_path = short_path1 + '-' + short_path
        name = os.path.splitext(short_path)[0]

        print(f"image_dir={image_dir}, short_path={short_path}, name={name}")
        
        self.add_header(name)
        ims, txts, links = [], [], []

        for label, im_data in visuals.items():
            im = utils.tensor2im(im_data)
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                #im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
                im = resize(im, (h, int(w * aspect_ratio)))
            if aspect_ratio < 1.0:
                #im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
                im = resize(im, (int(h / aspect_ratio), w))
                
            save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        self.add_images(ims, txts, links, width=width)



if __name__ == '__main__':  # we show an example usage here.
    html = HTML('web/', 'test_html', opt=None)
    html.add_header('hello world')

    ims, txts, links = [], [], []
    for n in range(4):
        ims.append('image_%d.png' % n)
        txts.append('text_%d' % n)
        links.append('image_%d.png' % n)
    html.add_images(ims, txts, links)
    html.save()
