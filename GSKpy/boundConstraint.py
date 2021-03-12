
def boundConstraint(vi,pop,low,high):

    # if the boundary constraint is violated, set the value to be the middle
# of the previous value and the bound
    
    #NP,D=size(pop,nargout=2)
# boundConstraint.m:8
    
    ## check the lower bound
    #xl=repmat(lu(1,arange()),NP,1)
# boundConstraint.m:11
    #pos=vi < low
# boundConstraint.m:13
    #vi[pos]=(pop(pos) + vi(pos)) / 2
    vi[vi < low] = ((pop + low)/2)[vi < low]
# boundConstraint.m:14
    ## check the upper bound
    #xu=repmat(lu(2,arange()),NP,1)
# boundConstraint.m:17
    #pos=vi > xu
# boundConstraint.m:18
    #vi[pos]=(pop(pos) + xu(pos)) / 2
    vi[vi > high] = ((pop + high)/2)[vi > high]
# boundConstraint.m:19