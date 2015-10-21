import decisiontree as dt

####################################################################################################
# Test get_best_feature()
####################################################################################################

print('\n\nTesting get_best_feature()');
print('----------------------------------------------------------------------------------------------------\n\n');

data = [
         [0,0,0],
         [1,1,1],
         [1,0,1],
         [0,1,0]
        ];
output = dt.get_best_feature(data, [0,1]);
print(output);
assert output == 0;

data = [
         [0,0,0],
         [0,1,1],
         [0,1,1],
         [0,0,0]
        ];
output = dt.get_best_feature(data, [0,1]);
print(output);
assert output == 1;

data = [
         [0,0,0,0],
         [0,0,1,1],
         [0,0,1,1],
         [0,0,0,0]
        ];
output = dt.get_best_feature(data, [0,1,2]);
print(output);
assert output == 2;

data = [
         [1,0,1],
         [1,1,0]
        ];
output = dt.get_best_feature(data, [1]);
print(output);
assert output == 1;

####################################################################################################
# Test entropy()
####################################################################################################

print('\n\nTesting entropy()');
print('----------------------------------------------------------------------------------------------------\n\n');

data = [0,0,0,0];
output = dt.entropy(data);
print(output);
assert output == 0;

data = [1,0,0,0];
output = dt.entropy(data);
print(output);
assert output != 0;

data = [1,0];
output = dt.entropy(data);
print(output);
assert output != 0;

####################################################################################################
# Test predict()
####################################################################################################

print('\n\nTesting predict()');
print('----------------------------------------------------------------------------------------------------\n\n');

idSubtrees = {0: 0, 1: 1};
notSubtrees = {0: 1, 1: 0};

data = [[1],[1],[0],[0]];
tree = 3;
output = dt.predict(tree, data);
print(output);
assert output == [3,3,3,3];

data = [[1],[1],[0],[0]];
tree = (0, {0: 10, 1:20});
output = dt.predict(tree, data);
print(output);
assert output == [20,20,10,10];

data = [[1,0],[1,0],[0,0],[0,0]];
tree = (0, {0: 10, 1:20});
output = dt.predict(tree, data);
print(output);
assert output == [20,20,10,10];

data = [[1,0],[1,0],[0,0],[0,0]];
tree = (1, {0: 10, 1:20});
output = dt.predict(tree, data);
print(output);
assert output == [10,10,10,10];

# AND function
data = [[0,0],[0,1],[1,0],[1,1]];
tree = (0, {0: 0, 1: (1, idSubtrees)});
output = dt.predict(tree, data);
print(output);
assert output == [0,0,0,1];

# XOR function
data = [[0,0],[0,1],[1,0],[1,1]];
tree = (0, {0: (1, idSubtrees), 1: (1, notSubtrees)});
output = dt.predict(tree, data);
print(output);
assert output == [0,1,1,0];

####################################################################################################
# Test build_decision_tree()
####################################################################################################

print('\n\nTesting build_decision_tree()');
print('----------------------------------------------------------------------------------------------------\n\n');

# Identity
data = [
         [0,0],
         [1,1]
        ];
feature_indices = [0];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == y;

# Not
data = [
         [0,1],
         [1,0]
        ];
feature_indices = [0];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == y;

# AND
data = [
         [0,0,0],
         [0,1,1],
         [1,0,1],
         [1,1,1]
        ];
feature_indices = [0,1];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == y;

# XOR
data = [
         [0,0,0],
         [0,1,1],
         [1,0,1],
         [1,1,0]
        ];
feature_indices = [0,1];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == y;

# Part of the XOR tree-building thing
data = [
         [1,0,1],
         [1,1,0]
        ];
feature_indices = [1];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == y;

# Multiple classes
data = [
         [0,0,2],
         [0,1,1],
         [1,0,3],
         [1,1,4]
        ];
feature_indices = [0,1];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == y;

# Data that can't be separated
data = [
         [0,0,1],
         [0,0,3],
         [0,0,3],
         [0,0,4]
        ];
feature_indices = [0,1];
tree = dt.build_decision_tree(data, feature_indices);
print(tree);
x = [d[:-1] for d in data];
y = [d[-1] for d in data];
output = dt.predict(tree, x);
print(output);
assert output == [3,3,3,3];


print('\n----------------------------------------------------------------------------------------------------');
print('Tests Successful!')
print('----------------------------------------------------------------------------------------------------\n');
