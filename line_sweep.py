import heapq
from line import *
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.animation import FuncAnimation


class line_Sweep():
    def __init__(self, **kwargs):
        self.init_status(sample_arrange())
        self.init_heap()

    def l(self,index):
        return self.lines[index][1]

    def idex(self, line):
        return self.lines[line][0]

    def init_status(self, lines):
        """ Returns a dictionary of the sorted lines, and that status datastructure"""
        ordered_lines = (sorted(lines, key = lambda x  : x[0]))
        self.lines =  [x for x in enumerate(ordered_lines)]
        self.status = [x for x in range(len(lines))]

    def init_heap(self):
        """ Takes a dict of lines an constructs the heap of intersection points
        
            Heap Values are in form: (x-coord, l1, l2)
        """
        self.heap = []
        for i in self.status[:-1]:
            (x, y) = find_inter(self.l(i), self.l(i+1))
            heapq.heappush(self.heap, (x, (i, i+1)))
        # print(self.heap)

    def removeInter(self, l1, l2):
        """ Takes two lines and removes their intersection in the heap"""
        # print("removing intersection between %d %d" % (l1, l2), self.heap)
        index = [idx for (idx, (i, (x, y))) in enumerate(self.heap) if ((x, y) == (l1, l2) or (x,y) == (l2, l1))]
        if index:
            i = index[0]
            # print("Intersection Present")
            self.heap[i] = self.heap[-1]
            self.heap.pop()
            if i < len(self.heap):
                heapq._siftup(self.heap, i)
                heapq._siftdown(self.heap, 0, i)

    def swap(self, l1, l2, loc):
        """ Takes two lines and swaps their location in the status and adds new intersections"""

        # TODO: https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
        #       remove intersections that are no longer next

        pos1, pos2 = self.idex(l1), self.idex(l2)
        self.status[pos1], self.status[pos2] = l2, l1
        self.lines[l1], self.lines[l2] = (pos2, self.l(l1)), (pos1, self.l(l2))

        if (pos1 != 0):
            self.removeInter(l1,self.status[pos1 - 1])
            a_line = self.status[pos1-1] # line above's number 
            (x, y) = find_inter(self.l(l2), self.l(a_line))
            if (x > loc):
                heapq.heappush(self.heap, (x, (a_line, l2)))
            
        if (pos2 != len(self.status) - 1):
            self.removeInter(l2,self.status[pos2 + 1])
            a_line = self.status[pos2+1] # line below's number
            (x, y) = find_inter(self.l(l1), self.l(a_line))
            if (x > loc):
                heapq.heappush(self.heap, (x, (l1, a_line)))

    def run(self):
        """ Runs lines sweep for an initialized arrangement """
        self.draw_lines()
        i = 0
        while self.heap:
            (x, (l1i, l2i)) = heapq.heappop(self.heap)  # l1i = line number
            (l1, l2) = (self.l(l1i), self.l(l2i))
            print("Intersection between lines: %d %d, at: (%f, %f)" % (l1i, l2i, x, y(l1,x)))
            # print(x, y(l1[1],x))
            self.swap(l1i, l2i, x)
            i += 1
        print(i)

    def iter_run_auto(self, i):
        """ Runs lines sweep for an initialized arrangement """
        if self.heap:
            (x, (l1i, l2i)) = heapq.heappop(self.heap)  # l1i = line number
            (l1, l2) = (self.l(l1i), self.l(l2i))
            print("Intersection between lines: %d %d, at: (%f, %f)" % (l1i, l2i, x, y(l1,x)))
            xs = []
            ys = []
            self.swap(l1i, l2i, x)
            for (xi, (l1, l2)) in self.heap:
                xs.append(xi)
                ys.append(y(self.l(l1), xi))
            points.set_data(xs, ys)
            vert.set_data(self.plotVert(x))
            return points
        else:
            return

    def iter_run(self, i):
        """ Runs lines sweep for an initialized arrangement """
        input("Press any key to continue...")
        if self.heap:
            (x, (l1i, l2i)) = heapq.heappop(self.heap)  # l1i = line number
            (l1, l2) = (self.l(l1i), self.l(l2i))
            print("Intersection between lines: %d %d, at: (%f, %f)" % (l1i, l2i, x, y(l1,x)))
            xs = []
            ys = []
            self.swap(l1i, l2i, x)
            for (xi, (l1, l2)) in self.heap:
                xs.append(xi)
                ys.append(y(self.l(l1), xi))
            points.set_data(xs, ys)
            vert.set_data(self.plotVert(x))
            return points
        else:
            return

    # Supporting Functions
    def bounding_region(self):
        """ Finds the max and min coordinates for intersections """
        res = [find_inter(a[1], b[1]) for idx, a in enumerate(self.lines) for b in self.lines[idx + 1:]]
        xl, xh = min(res, key=lambda x: x[0])[0], max(res, key=lambda x: x[0])[0]
        yl, yh = min(res, key=lambda x: x[1])[1], max(res, key=lambda x: x[1])[1]
        return (xl - 1, xh + 1, yl - 1, yh + 1)

    # Animation Section

    def plotVert(self, x):
        """ Plots a vertical line at the given x-coordate"""
        axis=plt.gca()
        x_vals = [x, x]
        y_vals = [-100, 100]
        return (x_vals, y_vals)

    def abline(self, slope, intercept):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--')

    def draw_lines(self):
        """ Plots all the lines from the line sweep"""
        print(self.lines)
        xl, xh, yl, yh = self.bounding_region()
        axis = plt.axes(xlim = (xl, xh), ylim = (yl, yh))
        for (n, (m, b)) in self.lines:
            self.abline(m, b)
        return axis

line = line_Sweep(update=1, augment=2)
fig = plt.figure()
axis = line.draw_lines()
points, = axis.plot([], [], 'ro')
vert, = axis.plot([], [], 'b-')

if (len(sys.argv) > 1):
    anim = FuncAnimation(fig, line.iter_run,
                frames=8,
                interval=100, 
                repeat=False)
    plt.show()
else:
    anim = FuncAnimation(fig, line.iter_run_auto, 
                frames=8,
                interval=100, 
                repeat=False)
    anim.save('./BaseProject/simple_arrange2.gif', writer = 'Pillow', fps = 5)
