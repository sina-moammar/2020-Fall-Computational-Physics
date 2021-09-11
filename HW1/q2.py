from automata import CA1D

sample = CA1D(201, [101])

# Rule = 110
sample.render(110, 200)
sample.show('q2-110')

# Rule = 75
sample.render(75, 200)
sample.show('q2-75')
