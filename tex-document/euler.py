def euler(f,x0,y0,h,n):
    x, y, result = x0, y0, [(x0,y0)]
    for i in range(n):
        y += f(x,y) * h
        x += h
        result.append((x,y))
    return result
