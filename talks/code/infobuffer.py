def rational_multiply(x, n, d, bitbuffer):
    bitbuffer.push(x % d, d)
    x /= d
    x *= n
    x += bitbuffer.pop(n)
    return x
