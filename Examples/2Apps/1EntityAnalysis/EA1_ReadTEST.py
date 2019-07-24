from D4M.assoc import *

# Read CSV file (take ~1 minute)
E = readcsv('Entity.csv')
E[0:10].printfull() # Print some rows in tabular form

# Extract columns
row, col, doc = E[:,'doc,'].find()
row, col, entity = E[:,'entity,'].find()
row, col, position = E[:,'position,'].find()
row, col, rowType = E[:,'type,'].find()

# Interleave type and entity strings
typeEntity = catstr(rowType,'|',entity)

# Create Sparse Edge Matrix
E = Assoc(doct, typeEntity, position)
E[0,:].printfull() # Show first row in tabular form
print(E) # show structure
E.transpose().spy() # Plot the transpose

# Write to a csv file
writecsv(E, 'E.csv')

