import heapq
import argparse
from line import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from math import comb
import random
from matplotlib.animation import FuncAnimation

from aug_functs import *

class aug_line_Sweep():
    """ The line sweep class
    
        Fields:
        lines (list)    : A list of the lines to sweep over
        init_aug (func) : The function to initialize the augmented status
        extra (object)  : The initial value of the extra variable
        update (func)   : The function to call at each step of the algo
        report (func)   : The function to call at the end to report
        aug_print (func): The function to print out the augments at each step
    """

    def __init__(self, **kwargs):
        """Constructor for the line sweep class
        
        Parameters:
        lines (list)    : A list of the lines to sweep over
        init_aug (func) : The function to initialize the augmented status
        extra (object)  : The initial value of the extra variable
        update (func)   : The function to call at each step of the algo
        report (func)   : The function to call at the end to report
        aug_print (func): The function to print out the augments at each step
        """
        if 'init_aug' in kwargs:
            self.update  = kwargs.get('update')
            self.init_aug = kwargs.get('init_aug')
            self.report = kwargs.get('report')
            self.aug_print = kwargs.get('aug_print')
            self.extra = 0
        else:
            self.update   = lambda x1, x2, x3, x4, x5, x6, x7: None, None
            self.init_aug = None
            self.report = lambda x1, x2, x3, x4: print("fin")
            self.extra = 0
            self.aug_print = lambda x1, x2: None
        if 'lines' in kwargs:
            self.init_status(kwargs.get('lines'))
        else:
            self.init_status(sample_arrange())
        self.init_heap()

    def l(self,index):
        """ Returns the line representation of a line at a given index"""
        return self.lines[index][1]

    def idex(self, line):
        """ Return a lines index in the status data structure"""
        return self.lines[line][0]

    def init_status(self, lines):
        """ Returns a dictionary of the sorted lines, and that status datastructure"""
        ordered_lines = (sorted(lines, key = lambda x  : x[0]))
        self.lines =  [x for x in enumerate(ordered_lines)]
        self.status = [x for x in range(len(lines))]
        if self.init_aug:
            self.aug_status, self.extra = self.init_aug(self.status, self.lines)
        else:
            self.aug_status = self.status * 0

    def init_heap(self):
        """ Takes a dict of lines an constructs the heap of intersection points
        
            Heap Values are in form: (x-coord, l1, l2)
        """
        self.heap = []
        for i in self.status[:-1]: # Finding intersections of adjacent lines
            (x, y) = find_inter(self.l(i), self.l(i+1))
            heapq.heappush(self.heap, (x, (i, i+1)))

    def removeInter(self, l1, l2):
        """ Takes two lines and removes their intersection in the heap

            Should be done in log time,
            however, it is not easily done with python
        """
        # TODO: https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
        #       remove intersections that are no longer next

        index = [idx for (idx, (i, (x, y))) in enumerate(self.heap) if ((x, y) == (l1, l2) or (x,y) == (l2, l1))]
        if index:
            i = index[0]
            self.heap[i] = self.heap[-1]
            self.heap.pop()
            if i < len(self.heap):
                heapq._siftup(self.heap, i)
                heapq._siftdown(self.heap, 0, i)

    def swap(self, l1, l2, loc):
        """ Takes two lines and swaps their location in the status and adds new intersections"""

        # finding their location in the status data structure
        pos1, pos2 = self.idex(l1), self.idex(l2)
        self.status[pos1], self.status[pos2] = l2, l1
        # updating the lines array with their new indices
        self.lines[l1], self.lines[l2] = (pos2, self.l(l1)), (pos1, self.l(l2))

        # if the first line was not the top of the list,
        #   its intersection with the line above must be removed
        #   and one must be added now that the other line in now next to it
        if (pos1 != 0):
            self.removeInter(l1,self.status[pos1 - 1])
            a_line = self.status[pos1-1] # line above's number 
            (x, y) = find_inter(self.l(l2), self.l(a_line))
            if (x > loc):
                heapq.heappush(self.heap, (x, (a_line, l2)))
        
        # the same logic applies for the other line being the bottom line
        if (pos2 != len(self.status) - 1):
            self.removeInter(l2,self.status[pos2 + 1])
            a_line = self.status[pos2+1] # line below's number
            (x, y) = find_inter(self.l(l1), self.l(a_line))
            if (x > loc):
                heapq.heappush(self.heap, (x, (l1, a_line)))

    def run(self):
        """ Runs lines sweep for an initialized arrangement """
        self.draw_lines()
        while self.heap: # iterate through all the intersection points
            (x, (l1i, l2i)) = heapq.heappop(self.heap)  # l1i = line number
            self.aug_status, self.extra = self.update(x, l1i, l2i, 
                self.status, self.lines, self.aug_status, self.extra)
            (l1, l2) = (self.l(l1i), self.l(l2i))
            self.aug_print(self.aug_status, self.extra)
            print("Intersection between lines: %d %d, at: (%f, %f)" % (l1i, l2i, x, y(l1,x)))
            self.swap(l1i, l2i, x)
        # once all the points are reached, report
        self.report(self.status, self.lines, self.aug_status, self.extra)

    def iter_run_auto(self, i):
        """ Runs lines sweep for an initialized arrangement """
        if self.heap: # checks to make sure the heap still has elements
            (x, (l1i, l2i)) = heapq.heappop(self.heap)  # l1i = line number
            self.aug_status, self.extra = self.update(x, l1i, l2i, 
                self.status, self.lines, self.aug_status, self.extra)
            (l1, l2) = (self.l(l1i), self.l(l2i))
            self.swap(l1i, l2i, x)
            
            # collecting the current points in the heap
            xs = []
            ys = []
            for (xi, (l1, l2)) in self.heap:
                xs.append(xi)
                ys.append(y(self.l(l1), xi))
            points.set_data(xs, ys)

            # drawing the current location of the sweep line
            vert.set_data(self.plotVert(x))

            return points
        else: # once the heap is empty just report
            self.report(self.status, self.lines, self.aug_status, self.extra)
            return

    def iter_run(self, i):
        """ Runs lines sweep for an initialized arrangement """
        input("Press enter to continue...") # waiting for user input to continue
        if self.heap: # checks to make sure the heap still has elements
            (x, (l1i, l2i)) = heapq.heappop(self.heap)  # l1i = line number
            self.aug_status, self.extra = self.update(x, l1i, l2i, 
                self.status, self.lines, self.aug_status, self.extra)
            (l1, l2) = (self.l(l1i), self.l(l2i))
            print("Intersection between lines: %d %d, at: (%f, %f)" % (l1i, 
                l2i, x, y(l1,x)))
            self.swap(l1i, l2i, x)

            # collecting the current points in the heap
            xs = []
            ys = []
            for (xi, (l1, l2)) in self.heap:
                xs.append(xi)
                ys.append(y(self.l(l1), xi))
            points.set_data(xs, ys)

            # drawing the current location of the sweep line
            vert.set_data(self.plotVert(x))
            
            # Printing the current status
            print("\tThe heap: ", self.heap)
            print("\tThe status array:", self.status)
            self.aug_print(self.aug_status, self.extra)
            return points
        else: # once the heap is empty just report
            self.report(self.status, self.lines, self.aug_status, self.extra)
            return

    # Supporting Functions
    def bounding_region(self):
        """ Finds the max and min coordinates for intersections """
        res = [find_inter(a[1], b[1])   for idx, a in enumerate(self.lines) 
                                        for b in self.lines[idx + 1:]]
        xl, xh = min(res, key=lambda x:x[0])[0], max(res, key=lambda x:x[0])[0]
        yl, yh = min(res, key=lambda x:x[1])[1], max(res, key=lambda x:x[1])[1]
        return (xl - 1, xh + 1, yl - 1, yh + 1)

    # Animation Section

    def plotVert(self, x):
        """ Plots a vertical line at the given x-coordate"""
        x_vals = [x, x]
        y_vals = [-10000, 10000]
        return (x_vals, y_vals)

    def abline(self, slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    def draw_lines(self):
        """ Plots all the lines from the line sweep"""
        xl, xh, yl, yh = self.bounding_region()
        axis = plt.axes(xlim = (xl, xh), ylim = (yl, yh))
        for (n, (m, b)) in self.lines:
            self.abline(m, b)
        return axis

def get_lines(num_lines):
    """ Returns a list of randomly generated lines """
    slopes = random.sample(range(-2*num_lines, 2*num_lines), num_lines)
    inters = random.sample(range(-2*num_lines, 2*num_lines), num_lines)
    return list(zip(slopes, inters))

# Configuring the parse
parser = argparse.ArgumentParser(description='An augmented line sweep algorithm')
parser.add_argument('-m', help='manual mode', action='store_true')
parser.add_argument('-l', help='num_lines', type=int, default=4)
parser.add_argument('-o', help='out_file', type=str, default='out.gif')
args = parser.parse_args()

# Creating the line sweep
line = aug_line_Sweep(update=stat_upd, init_aug=aug_stat_init, 
                    report=aug_report, aug_print=aug_print,
                    lines=get_lines(args.l))

# Figure setup
fig = plt.figure()
axis = line.draw_lines()
points, = axis.plot([], [], 'ro')
vert, = axis.plot([], [], 'b-')

# run mode
if args.m: # manual mode, with breaks
    anim = FuncAnimation(fig, line.iter_run,
                frames=comb(args.l, 2) + 1,
                init_func=lambda : ...,
                interval=100, 
                repeat=False)
    plt.show()
else:
    anim = FuncAnimation(fig, line.iter_run_auto, 
                frames=comb(args.l, 2) + 1,
                init_func=lambda : ...,
                interval=100, 
                repeat=False)
    anim.save(args.o, writer = 'Pillow', fps = 5)
