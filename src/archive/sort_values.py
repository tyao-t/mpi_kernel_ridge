import pandas as pd

train_data = pd.read_csv('data/housing_train.tsv', sep='\t', header=None)
test_data = pd.read_csv('data/housing_test.tsv', sep='\t', header=None)

sorted_train_data = train_data.sort_values(by=train_data.columns[-1])
sorted_test_data = test_data.sort_values(by=test_data.columns[-1])

sorted_train_data.to_csv('sorted_train.tsv', sep='\t', header=None, index=False)

sorted_test_data.to_csv('sorted_test.tsv', sep='\t', header=None, index=False)

x_test = sorted_train_data.iloc[:, :-1].values
y_test = sorted_train_data.iloc[:, -1].values.reshape(-1, 1)
n = x_test.shape[0]
# print(n)

d = {}
for i in range(5): d[i] = []
sum = 0
for i in range(n-int(n*0.3), n): sum += x_test[i][5]/x_test[i][6]
print(sum / int(n*0.3))
# print(d)
# for i in range(5): print(sum(d[i])/len(d[i]))

# for i in range(n): 
#     if x_test[i][8] == 4: print(y_test[i])



# longitude: A measure of how far west a house is; a higher value is farther west
# latitude: A measure of how far north a house is; a higher value is farther north
# housingMedianAge: Median age of a house within a block; a lower number is a newer building
# totalRooms: Total number of rooms within a block
# totalBedrooms: Total number of bedrooms within a block
# population: Total number of people residing within a block
# households: Total number of households, a group of people residing within a home unit, for a block
# medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
# oceanProximity: Location of the house w.r.t ocean/sea (0 = <1H OCEAN, 1 = INLAND, 2 = NEAR OCEAN, 3 = NEAR BAY)
# medianHouseValue: Median house value for households within a block (measured in US Dollars)