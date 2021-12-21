from line import *

def dist(l1, l2, x):
    return y(l2,x) - y(l1,x)

def area(l1, l2, l3):
    x1, y1 = l1
    x2, y2 = l2
    x3, y3 = l3
    
    return abs(((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) + 0. )/ 2)

def stat_upd(x, l1, l2, status, lines, aug_status, extra):
    top_index = lines[l1][0]
    if top_index != 0:
        top_dist = dist(lines[l1][1], lines[status[top_index-1]][1], x)
    else:
        top_dist = float('inf')
    if (top_index + 2 < len(lines)):
        bot_dist = dist(lines[l2][1], lines[status[top_index+2]][1], x)
    else:
        bot_dist = float('inf')

    if (top_dist < bot_dist):
        index = status[top_index-1]
    else:
        index = status[top_index-1]
    A = area(lines[l1][1], lines[l2][1], lines[index][1])
    if A < extra[0]:
        return aug_status, (A, l1, l2, index)
    else:
        return aug_status, extra

def aug_stat_init(status, lines):
    return (status, (float('inf'), 0, 0, 0))

def aug_report(x1, x2, x3, x4):
    print("The minimum triangle is formed by the lines: %d, %d, %d, with area: %f" % (x4[1], x4[2], x4[3], x4[0]))

def aug_print(aug_status, extra):
    print("\tThe minimum triangle so far has area: %f, and is formed by lines %d, %d, and %d" % (extra[0], extra[1], extra[2], extra[3]))
