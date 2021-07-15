import numpy as np
import os
import sys
import ntpath
import time
from subprocess import Popen, PIPE
from pathlib import Path

from common import utils
from common.vis.html import HTML
from common.image import tensor2im, save_image
from common.landmark import landmark_to_image


class Visualizer():
    """This class includes several functions that can display/save images 
    and print/save logging information.

    It uses a Python library 'visdom' for display, 
    and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, logger):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt            # cache the option
        self.logger = logger      # logger info

        self.name = opt.MODEL.name
        self.path_output = Path(opt.OUTPUT_DIR)        # the output path for debug image 
        
        if self.opt.VISUAL.use_visdom:  # connect to a visdom server given <vsdm_port> and <vsdm_server>
            import visdom
            self.ncols = opt.VISUAL.vsdm_ncol
            self.vis = visdom.Visdom(server=opt.VISUAL.vsdm_server, port=opt.VISUAL.vsdm_port, env=opt.VISUAL.vsdm_env)
            
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.opt.VISUAL.use_HTML:  
            # create an HTML object at <checkpoints_dir>/web/; 
            # images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = self.path_output.joinpath('web')
            self.img_dir = self.path_output.joinpath('web', 'images')

            print(f'create web directory {self.web_dir} ...')
            self.img_dir.mkdir(parents=True, exist_ok=True)
            

        # create a logging file to store training losses
        self.logger.info(f'================ Training Loss ({time.strftime("%c")}) ================\n')

    def reset(self):
        """Reset the self.saved status"""
        pass

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, 
        this function will start a new server at port < self.opt.VISUAL.vsdm_port > """
        
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.opt.VISUAL.vsdm_port
        self.logger.info('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        self.logger.info('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)


    def display_current_results(self, visuals, epoch=0, batch=0):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        
        if self.opt.VISUAL.use_visdom:  # show images in the browser using visdom
            ncols = self.ncols
            
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                
                table_css = """<style>
                    table {border-collapse: separate; 
                           border-spacing: 4px; 
                           border-left-color: red;
                           white-space: nowrap; 
                           text-align: center}
                    table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                    </style>""" % (w, h)  # create a table css
                
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                
                try:
                    self.vis.images(images, nrow=ncols, win=self.opt.VISUAL.vsdm_id + 1, padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.opt.VISUAL.vsdm_id + 2, opts=dict(title=title + ' labels'))
                
                except Exception:
                    print("CANNOT vis image, text")
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label), win=self.opt.VISUAL.vsdm_id + idx)
                        idx += 1
                except Exception:
                    print("CANNOT vis image")
                    self.create_visdom_connections()

        if self.opt.VISUAL.use_HTML:  # save images to an HTML file 
            
            # create a webpage
            webpage = HTML(str(self.web_dir), 'Experiment name = %s' % self.name, opt=self.opt, refresh=-1)

            # save images to the disk
            for ilabel, data in visuals.items():
                image_numpy = tensor2im(data)
                img_path = self.img_dir.joinpath(f'epoch_{epoch:04d}_{batch:04d}_{ilabel}.png')
                save_image(image_numpy, img_path)

            # for epoch and batch 
            for iepoch in range(epoch, 0, -1):
                # print(f"DEBUG: iepoch={iepoch}, batch={batch}")
                if iepoch % self.opt.FREQ.epoch_display != 0 : continue
                
                for ibatch in range(batch, -1, -1):
                    # print(f"DEBUG: ibatch={ibatch}")
                    if ibatch % self.opt.FREQ.batch_display != 0: continue

                    # print(f"DEBUG add page: iepoch={iepoch} ibatch={ibatch}")
                    webpage.add_header(f'epoch [{iepoch:04d}] - Batch [{ibatch:04d}]')
                    ims, txts, links = [], [], []

                    for ilabel in visuals.keys():
                        img_path = f'images/epoch_{iepoch:04d}_{ibatch:04d}_{ilabel}.png'
                        ims.append(img_path)
                        txts.append(ilabel)
                        links.append(img_path)

                    webpage.add_images(ims, txts, links, width=self.opt.VISUAL.vsdm_wsize)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not self.opt.VISUAL.use_visdom: return

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        
        try:
            # print("np.array(self.plot_data['X'])=", np.array(self.plot_data['X']))
            # print("np.array(self.plot_data['Y'])=", np.array(self.plot_data['Y']))
            # print("self.plot_data['legend']", self.plot_data['legend'])


            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.opt.VISUAL.vsdm_id)
        except Exception:
            raise("CANNOT vis line")
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp=None, t_data=None):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        msg = f"[Epoch:{epoch:03d}-{iters:03d}]" 
        for k,v in losses.items():
            msg += f" {k}= {v:0.5f} " 

        self.logger.info(f'{msg}')  # save the message
