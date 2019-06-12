from D4M.assoc import *

# ## Creating Associative Arrays

# Hello! This is a test on the basic Assoc Array construction.
# Associative Array takes on entries of triplets, and it will parse an array of substrings
# that is divided by char divider.
# --Please note that this divider is indicated as the last char in the string.

row = "a,a,a,a,a,a,a,aa,aaa,b,bb,bbb,a,aa,aaa,b,bb,bbb,"
column = "a,aa,aaa,b,bb,bbb,a,a,a,a,a,a,a,aa,aaa,b,bb,bbb,"
values = "a-a,a-aa,a-aaa,a-b,a-bb,a-bbb,a-a,aa-a,aaa-a,b-a,bb-a,bbb-a,a-a,aa-aa,aaa-aaa,b-b,bb-bb,bbb-bbb,"

# Create assoc array and list triples
A = Assoc(row, column, values)

# This is the data structure of the Associative Array Class

print(A)

# The Assoc.printfull() method allows it to be printed in a tabular form.

A.printfull()

# When written into CSV form, the data is stored in the tabular form

writecsv(A, "A.csv")
