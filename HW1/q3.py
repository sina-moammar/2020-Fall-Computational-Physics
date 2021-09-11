from automata import CA1D

sample = CA1D(201, [101])

# Rule = 46
sample.render(46, 300)
sample.show('q3-46')

# Rule = 78
sample.render(78, 300)
sample.show('q3-78')

# Rule = 102
sample.render(102, 300)
sample.show('q3-102')

# Rule = 106
sample.render(106, 300)
sample.show('q3-106')

# Rule = 108
sample.render(108, 300)
sample.show('q3-108')

# Rule = 111
sample.render(111, 300)
sample.show('q3-111')

# Rule = 126
sample.render(126, 300)
sample.show('q3-126')

# Rule = 238
sample.render(238, 300)
sample.show('q3-238')


### Random initialization ###
sample = CA1D(201, 'rand')

# Rule = 46
sample.render(46, 200)
sample.show('q3-46-rand')

# Rule = 78
sample.render(78, 200)
sample.show('q3-78-rand')

# Rule = 102
sample.render(102, 200)
sample.show('q3-102-rand')

# Rule = 106
sample.render(106, 200)
sample.show('q3-106-rand')

# Rule = 108
sample.render(108, 200)
sample.show('q3-108-rand')

# Rule = 111
sample.render(111, 200)
sample.show('q3-111-rand')

# Rule = 126
sample.render(126, 200)
sample.show('q3-126-rand')

# Rule = 238
sample.render(238, 200)
sample.show('q3-238-rand')

# Rule = 110
sample.render(110, 200)
sample.show('q3-110-rand')

# Rule = 75
sample.render(75, 200)
sample.show('q3-75-rand')
