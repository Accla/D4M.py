from D4M.assoc import *
import D4M.util as util

# Read CSV file (take ~1 minute)
E = readcsv('Entity.csv')
E[0:10, :].printfull()  # Print some rows in tabular form

# Extract columns
_, _, doc = E[:, 'doc,'].find()
_, _, entity = E[:, 'entity,'].find()
_, _, position = E[:, 'position,'].find()
_, _, rowType = E[:, 'type,'].find()

# Interleave type and entity strings
typeEntity = util.catstr(rowType, entity, separator='|')

# Create Sparse Edge Matrix
E = Assoc(doc, typeEntity, position)
E[0, :].printfull()  # Show first row in tabular form
print(E)  # show structure
E.transpose().spy()  # Plot the transpose

# Write to a csv file
writecsv(E, 'E.csv')
