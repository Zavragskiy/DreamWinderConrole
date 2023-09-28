def getkb(x1, x2, y1, y2):
    k = (y2-y1)/(x2-x1)
    b = y1 - k*x1
    return [k ,b]

def CharLine(Char = '', w = 20.0):
    h = 2 * w
    charLine = [[], []]
    if Char == ' ':
        return charLine
    elif Char == '-':
        charLine = [[0.0, w  ],
                    [h/2, h/2]]
    elif Char == '1':
        charLine = [[w,   w, 0.0],
                    [0.0, h, h/2]]
    elif Char == '2':
        charLine = [[w,   0.0, w,     w, 0.0, 0.0],
                    [0.0, 0.0, h / 2, h, h,   h/2]]
    elif Char == '3':
        charLine = [[0.0, w,   w,   0.0, w, 0.0],
                    [0.0, 0.0, h/2, h/2, h, h  ]]
    elif Char == '4':
        charLine = [[w,   w,   w,   0.0, 0.0],
                    [0.0, h,   h/2, h/2, h, ]]
    elif Char == '5':
        charLine = [[0.0, w,   w,   0.0, 0.0, w],
                    [0.0, 0.0, h/2, h/2, h,   h]]
    elif Char == '6':
        charLine = [[0.0, w,   w,   0.0, 0.0, w],
                    [h/2, h/2, 0.0, 0.0, h/2, h]]
    elif Char == '7':
        charLine = [[w/2, w, 0.0, 0.0    ],
                    [0.0, h, h  , h * 3/4]]
    elif Char == '8':
        charLine = [[w,   w,   0.0, 0.0, w,   w, 0.0, 0.0],
                    [h/2, 0.0, 0.0, h/2, h/2, h, h,   h/2]]
    elif Char == '9':
        charLine = [[0.0, w,   w, 0.0, 0.0, w, ],
                    [0.0, h/2, h, h,   h/2, h/2]]
    elif Char == '0':
        charLine = [[0.0, w,   w, 0.0, 0.0],
                    [0.0, 0.0, h, h,   0.0]]

    else:
        return charLine
    return charLine