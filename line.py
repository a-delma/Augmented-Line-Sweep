def line(m, b):
    """ Constructor for a line, maintains a unique conter"""
    return (m, b)

def y(l,x):
    """ Returns the y coordinate of the line at the given x"""
    return l[0] * x + l[1]

def find_inter(l1, l2):
    """ Returns the (x, y) of the two lines"""

    # allows for the two representations of the lines to be passed
    if isinstance(l1[1], tuple):
        (_, (m1, b1)) = l1
        (_, (m2, b2)) = l2
    else:
        (m1, b1) = l1
        (m2, b2) = l2

    # checks to see if they are parallel or the same line
    if (m1 == m2):
        if (b1 == b2):
            return (0, b1)
        return False # we dont allow for parallel lines here
    return ((b2 - b1) / (m1 - m2), m2 * (b2 - b1) / (m1 - m2) + b2)

def sample_arrange():
    """ A sample arrangement of lines"""
    l1 = line( 1,  1)
    l2 = line( 2, -1)
    l3 = line( 3,  1)
    l4 = line( 4,  5)
    l5 = line(-5,  8)
    return [l1, l2, l3, l4, l5]

def basic_arrange():
    """ A basic arrangement of lines"""
    l1 = line(1, 1)
    l2 = line(2, -1)
    l3 = line(3, 1)
    return [l1, l2, l3]