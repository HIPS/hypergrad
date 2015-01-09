Investigating the wiggliness of the loss as a function of the scale of
the initial parameters.

![](1/fig.png)

The actual loss looks very smooth but the gradient still goes
crazy. It could be numerical instability or it could conceivably be
that the apparent smoothness goes away on a sufficiently small length
scale. Testing this would be hard because numerical error will also
make it look rough on a small length scale. I think you'd want to try
looking for smooth oscillations. There's a hint of this at the 10
iteration level. Let's look at this closer, from, say, -2 to 1....

![](2/fig.png)

Ok, I think it's pretty clear that the function gets wiggly. But maybe
we would be ok if we stayed on the peaceful side of chaos? Let's try
more iterations...

![](3/fig.png)

Very interesting. It really does look like the chaos chills right out
when the parameter scale is small enough.
