import os
import math
import random
import numpy as np

class Dataset:
    dataset_filename = ""
    entries = []
    columns = []
    column_types = []
    filtered_entries = []
    feature_list = []
    feature_types = []
    feature_columns = []
    feature_min_maxes = []
    class_list = []
    class_types = []
    class_columns = []
    
    nan_values = {}

    should_normalize = False

    prenorm_feature_min_maxes = []
    prenorm_features_mean = []
    prenorm_features_variance = []

    K = 0
    N = 0

    DATASET_FOLDER = ""

    def __init__(self, DATASET_FOLDER=""):
        self.DATASET_FOLDER = DATASET_FOLDER

        # Loads dataset and entries
        while True:
            self.dataset_filename = input("Dataset filename (within the folder f`{DATASET_FOLDER}`): ")
            if os.path.isfile(DATASET_FOLDER + self.dataset_filename):
                self.populate_entries()
                print("Dataset loaded.")
                break
            else:
                print("Error: Invalid file path. File was not found.")

        # (If chosen) Filters the entries with the given parameters
        filtered_entries = self.entries
        while True:
            should_filter = input("Filter samples (y/n)? ")
            if should_filter == "y":
                should_filter = True
            elif should_filter == "n":
                should_filter = False
            else:
                print("Invalid input. Should be 'y' or 'n' without the quotation marks.")
                continue

            filtered_entries = self.filter_entries(should_filter, filtered_entries)
            if not should_filter: break

        self.filtered_entries = filtered_entries

        print(f"Total samples: {len(self.entries)}")
        print(f"Total filtered samples: {len(self.filtered_entries)}")

        '''
        self.N = len(self.filtered_entries)

        result = self.get_dataset_stats(True)
        self.features_mean = result["features_mean"]
        self.features_variance = result["features_variance"]
        self.features_std = result["features_std"]
        self.dependent_variables_mean = result["dependent_variables_mean"]
        self.dependent_variables_variance = result["dependent_variables_variance"]
        self.dependent_variables_std = result["dependent_variables_std"]

        self.should_normalize = False
        normalization_warned = False
        while True:
            if not normalization_warned:
                print("\nNote: Normalizing the dataset for training results in weights trained on the normalized values. (Meaning different than those weights on non-normalized samples)")
                print("However, it sometimes tends to make data processing faster, among other things.")
                print("The model should perform similarly given the ideal parameters")
                print("Restricts training data to [-1, 1]")
                print("The weights lose some interpretation value compared to training on non-normalized data, if you're trying to infer things from looking at them as is.")
                print("i.e. In a model that predicts house prices with respect to its area (m²), the feature weight B1 represents the price per square meter of the model. However, if normalized, you would have to perform some additional calculations in order to get that number back, as it would need to be 'denormalized'.")
                
                normalization_warned = True

            
            should_normalize = input("Normalize dataset? (y/n) ")

            if should_normalize == 'y':
                self.should_normalize = True
            elif should_normalize == 'n':
                self.should_normalize = False
            else:
                print("Invalid input. Should be 'y' or 'n'.")
                continue
            
            break

        self.feature_min_maxes = []
        for feature_type in self.feature_types:
            if feature_type == "str":
                self.feature_min_maxes.append([0, -math.inf])
            elif feature_type == "float":
                self.feature_min_maxes.append([math.inf, -math.inf])

            # Mean normalization (sets mean of feature axes to 0) & variance normalization

        for entry in self.filtered_entries:
            for feature_idx, feature in enumerate(self.feature_list):
                entry_old_value = entry.features[feature]

                if self.should_normalize:
                    entry.features[feature] -= self.features_mean[feature_idx]

                    if entry == self.filtered_entries[0]:
                        print(f"Original: {self.filtered_entries[0].column_value_dict}")
                        print(f"After mean normalization: {self.filtered_entries[0].features}")

                    # entry.features[feature] /= self.features_variance[feature_idx]
                    entry.features[feature] /= self.features_std[feature_idx]

                    if entry == self.filtered_entries[0]:
                        print(f"After variance normalization: {self.filtered_entries[0].features}")
                        

                if self.feature_types[feature_idx] == "str":
                    for key in self.nan_values:
                        if self.nan_values[key] == entry_old_value:
                            self.nan_values[key] = entry.features[feature]

                self.feature_min_maxes[feature_idx][0] = min(entry.features[feature], self.feature_min_maxes[feature_idx][0])
                self.feature_min_maxes[feature_idx][1] = max(entry.features[feature], self.feature_min_maxes[feature_idx][1])

        if self.should_normalize:
            
            # Normalizes features to [-1, 1] range
            for entry in self.filtered_entries:
                if entry == self.filtered_entries[0] and self.should_normalize:
                    print(f"Before -1, 1: {entry.features}")
                    self.normalize(entry)
                    print(f"After -1, 1: {entry.features}")
                else:
                    self.normalize(entry)

            result = self.get_dataset_stats(True)
            self.features_mean = result["features_mean"]
            self.features_variance = result["features_variance"]
            self.features_std = result["features_std"]
            self.dependent_variables_mean = result["dependent_variables_mean"]
            self.dependent_variables_variance = result["dependent_variables_variance"]
            self.dependent_variables_std = result["dependent_variables_std"]
        '''
  
    def populate_entries(self):
        print("Note: The dataset must be .csv formatted, with ',' as a delimiter, and each column in the first row must contain the respective feature name.")
        print("i.e.:")
        print("id,area,price")
        print("0,100,400000")
        print("1,50,250000")
        print("2,200,900000\n")

        with open(self.DATASET_FOLDER + self.dataset_filename, "r", encoding="utf-8") as dataset:
            rows = dataset.readlines()
            column_names = rows[0].replace("\n", "").split(",")
            columns = column_names
            self.column_types = [None for _ in range(len(columns))]
            print(f"There are {len(column_names)} columns. These are the column names, in ascending column order: {', '.join(column_names)}.")
            print(f"WARNING: Currently, when training models with a dependent variable that is a string, you will only be predict values with it once (after it finishes training), this is because a way to store and load dictionaries for string<->number conversions are yet to be implemented. The model architecture and weights can still be saved however. Just keep in mind that it won't work if you try to load it to make predictions.")
            column_warning_printed = False
            name_warning_printed = False

            column_types_saved = False

            # Breaks loop if all available features have been chosen, as well as the dependent variable
            while len(self.feature_list) < len(column_names) - 1 or len(self.class_list) < len(column_names) - 1:

                if len(self.feature_list) > 0 and len(self.class_list) > 0:
                    result = input("Select more features/classes (y/n)? ")
                    if result == "n":
                        break
                    elif result != "y":
                        print("Invalid input. Must be 'y' or 'n'.")
                        continue

                # Selects a dependent or independent feature by column name or number
                # Does some validation checks (not enough, though)
                while True:
                    result = input("Select feature/class by column name or column number (0 for name / 1 for number)? ")
                    if result != "0" and result != "1":
                        print("Invalid input. Must be '0' or '1'.")
                        continue
                    
                    # Selects by column name
                    elif result == "0":
                        if not name_warning_printed:
                            print(f"The column name is the exact string in that column.")
                            name_warning_printed = True

                        name = input("Column name: ")
                        if name not in column_names:
                            print(f"Invalid column name: {name}. Valid column names: {', '.join(column_names)}")
                            continue

                        if name in self.feature_list:
                            print(f"Error: This column ({name}) is already in the feature list.")
                            continue

                        if name in self.class_list:
                            print(f"Error: This column ({name}) is already in the class list.")
                            continue

                        data_type = input("Column data type ('str' or 'float'): ").strip()
                        if data_type not in ['str', 'float']:
                            print(f"Invalid data type: {data_type}. Must be one of 'str', or 'float'.")
                            continue

                        dependent_or_independent = input("Dependent or independent feature (d/i)? ").strip()
                        if dependent_or_independent not in ['d', 'i']:
                            print(f"Invalid feature type: {dependent_or_independent}. Must be 'd' for dependent variable, or 'i' for independent variable.")
                            continue

                        number = column_names.index(name)

                        if dependent_or_independent == 'i':
                            self.feature_list.append(name)
                            self.feature_types.append(data_type)
                            self.feature_columns.append(number)
                        elif dependent_or_independent == 'd':
                            self.class_list.append(name)
                            self.class_types.append(data_type)
                            self.class_columns.append(number)

                    # Selects by column number
                    elif result == "1":
                        # Check whether the name is not unique and if so prompt user for the column number

                        if not column_warning_printed:
                            print("WARNING! FOR THIS TO WORK PROPERLY, EACH COLUMN NAME MUST BE UNIQUE.")
                            print(f"The column number is an integer in the range [0, {len(column_names) - 1}].")
                            print("You cannot choose an already chosen column number, as it leads to redundancy and linear dependence.")
                            print("Exactly one column must be the dependent feature.")
                            column_warning_printed = True
                        # add by number

                        number = input("Column number: ").strip()
                        try:
                            number = int(number)
                        except:
                            print(f"Invalid column number: {number}. Must be an integer.")
                            continue

                        if number < 0 or number > len(column_names) - 1:
                            print(f"Invalid column number: {number}. Must be in the range [0,{len(column_names) - 1}]")
                            continue

                        if number in self.feature_columns:
                            print(f"Error: This column ({number}) is already in the feature list.")
                            continue

                        if number == self.class_columns:
                            print(f"Error: This column ({name}) is already in the class list.")
                            continue

                        name = column_names[number]

                        data_type = input("Column data type ('str' or 'float'): ").strip()
                        if data_type not in ['str', 'float']:
                            print(f"Invalid data type {data_type}. Must be one of 'str' or 'float'.")
                            continue

                        dependent_or_independent = input("Dependent or independent feature (d/i)? ")
                        if dependent_or_independent not in ['d', 'i']:
                            print(f"Invalid feature type: {dependent_or_independent}. Must be 'd' for dependent variable, or 'i' for independent variable.")
                            continue
                        

                        if dependent_or_independent == 'i':
                            self.feature_list.append(name)
                            self.feature_types.append(data_type)
                            self.feature_columns.append(number)
                        elif dependent_or_independent == 'd':
                            self.class_list.append(name)
                            self.class_types.append(data_type)
                            self.class_columns.append(number)
                    
                    break

            print("Finished selecting features.")               

            for line in rows[1:]:
                try:
                    entry_columns = line.replace("\n", "").split(",")

                    feature_values = []
                    for feature_type, feature_column in zip(self.feature_types, self.feature_columns):
                        value = entry_columns[feature_column]
                        if value.strip() == '':
                            raise ValueError("Invalid feature value found, entry will be skipped")
                                             
                        match feature_type:
                            case "float":
                                value = float(value)
                            case "str":
                                if value not in self.nan_values.keys():
                                    self.nan_values[value] = random.randint(1, len(rows))

                                value = self.nan_values[value]

                        feature_values.append(value)

                    dependent_values = []
                    for dependent_variable_type, dependent_variable_column in zip(self.class_types, self.class_columns):
                        dependent_variable_value = entry_columns[dependent_variable_column]
                        if dependent_variable_value.strip() == '':
                            raise ValueError("Invalid class value found, entry will be skipped")
                    
                        match dependent_variable_type:
                                case "float":
                                    dependent_variable_value = float(dependent_variable_value)
                                case "str":                                   
                                    if dependent_variable_value not in self.nan_values.keys():
                                        self.nan_values[dependent_variable_value] = random.randint(1, len(rows))

                                    dependent_variable_value = self.nan_values[dependent_variable_value]
                        
                        dependent_values.append(dependent_variable_value)


                    # Note: The following code section stores all column/value pairs for the current entry, even if their value is empty/not given (opposed to the selection for the feature and dependent variable values above)
                    # If the column is equal to 1 or "1", the script will always treat it as the number 1.
                    column_value_dict = {}
                    for idx, column_name in enumerate(column_names):

                        # Value is considered as string type if failed to convert to number
                        column_value = entry_columns[idx]
                        column_type = 'str'
                        try:
                            column_value = float(column_value)
                            column_type = 'float'
                        except:
                            pass

                        if not column_types_saved:
                            self.column_types[idx] = column_type

                            if idx == len(column_names) - 1:
                                column_types_saved = True

                        column_value_dict[column_name] = column_value

                    new_entry = Entry(self, feature_values, dependent_values, column_value_dict)
                    self.entries.append(new_entry)
                except:
                    continue
            
            dataset.close()

        print("Finished populating the entries.")

    def custom_filter(self, entry, target_features, comparison_targets, comparison_operators):
        for feature, target, operator in zip(target_features, comparison_targets, comparison_operators):
            cmp = True

            if feature != "distance_builtin":
                match operator:
                    case '==':
                        if feature in self.class_list:
                            cmp = entry.dependent_values[feature] == target
                        elif feature in self.feature_list:
                            cmp = entry.features[feature] == target
                        else:
                            cmp = entry.column_value_dict[feature] == target
                    case '!=':
                        if feature in self.class_list:
                            cmp = entry.dependent_values[feature] != target
                        elif feature in self.feature_list:
                            cmp = entry.features[feature] != target
                        else:
                            cmp = entry.column_value_dict[feature] != target
                    case '<':
                        if feature in self.class_list:
                            cmp = entry.dependent_values[feature] < target
                        elif feature in self.feature_list:
                            cmp = entry.features[feature] < target
                        else:
                            cmp = entry.column_value_dict[feature] < target
                    case '<=':
                        if feature in self.class_list:
                            cmp = entry.dependent_values[feature] <= target
                        elif feature in self.feature_list:
                            cmp = entry.features[feature] <= target
                        else:
                            cmp = entry.column_value_dict[feature] <= target
                    case '>=':
                        if feature in self.class_list:
                            cmp = entry.dependent_values[feature] >= target
                        elif feature in self.feature_list:
                            cmp = entry.features[feature] >= target
                        else:
                            cmp = entry.column_value_dict[feature] >= target
                    case '>':
                        if feature in self.class_list:
                            cmp = entry.dependent_values[feature] > target
                        elif feature in self.feature_list:
                            cmp = entry.features[feature] > target
                        else:
                            cmp = entry.column_value_dict[feature] > target

            if not cmp:
                return False

        return True

    def filter_entries(self, should_filter, filtered_list):
        if should_filter:
            print("Note: The filtering occurs only once, for all features.")

            valid_operators = ["==", "!=", "<", "<=", ">=", ">"]
            target_features = []
            comparison_targets = []
            comparison_operators = []
            while True:
                l_target_features = input(f"Which features/columns to filter (comma-separated and no whitespace i.e. 'area_m2,price_brl')? Options: {', '.join(self.columns)} ")

                l_target_features = l_target_features.split(",")

                target_invalid = False

                for target_feature in l_target_features:
                    if target_feature not in self.feature_list and target_feature not in self.class_list and target_feature != "distance_builtin" and target_feature not in self.columns:
                        print(f"Error: Could not find feature {target_feature}")
                        target_invalid = True
                        break

                if target_invalid:
                    continue

                l_comparison_targets = []
                
                # These checks are separated in different loops (resulting in 3 times the necessary time for execution)
                # The reason is for more clarity and isolation of validation checks
                # It shouldn't slow down most times, as there usually aren't that many features (at least with what I have in mind)
                # Also, it is not optimized and I could definitely write this in a more isolated, clear way, with faster runtimes as well
                for target_feature in l_target_features:
                    if target_feature in self.feature_list:
                        feature_index = self.feature_list.index(target_feature)
                        feature_type = self.feature_types[feature_index]

                        comparison_target = input(f"Type in the target value to compare the column '{target_feature}' against. Your input must be a valid '{feature_type}'! ")

                        if feature_type == "str" and comparison_target not in self.nan_values.keys():
                            print(f"Error: Invalid target value. Given value '{comparison_target}' is not in the dataset.")
                            target_invalid = True
                            break
                        elif feature_type == "float":
                            try:
                                comparison_target = float(comparison_target)
                            except:
                                print(f"Error: Invalid target value. Could not convert '{comparison_target}' to a number.")
                                target_invalid = True
                                break

                        l_comparison_targets.append(comparison_target)
                    elif target_feature in self.class_list:
                        class_index = self.class_list.index(target_feature)
                        class_type = self.class_types[class_index]

                        comparison_target = input(f"Type in the target value to compare the column '{target_feature}' against. Your input must be a valid '{class_type}'! ")

                        if class_type == "str" and comparison_target not in self.nan_values.keys():
                            print(f"Error: Invalid target value. Given value '{comparison_target}' is not in the dataset.")
                            target_invalid = True
                            break
                        elif class_type == "float":
                            try:
                                comparison_target = float(comparison_target)
                            except:
                                print("Error: Invalid target value. Could not convert it to a number.")
                                target_invalid = True
                                break

                        l_comparison_targets.append(comparison_target)
                    else:
                        column_index = self.columns.index(target_feature)
                        column_type = self.column_types[column_index]

                        comparison_target = input(f"Type in the target value to compare the column '{target_feature}' against. Your input must be a valid '{column_type}'! ")

                        if column_type == "float":
                            try:
                                comparison_target = float(comparison_target)
                            except:
                                print("Error: Invalid target value. Could not convert it to a number.")
                                target_invalid = True
                                break

                        l_comparison_targets.append(comparison_target)


                if target_invalid:
                    continue

                l_comparison_operators = []
                invalid_operator = False

                for idx, target_feature in enumerate(l_target_features):
                    if target_feature in self.feature_list:
                        feature_index = self.feature_list.index(target_feature)
                        feature_type = self.feature_types[feature_index]
                    
                        comparison_operator = input(f"Type in the comparison operator to compare the column '{target_feature}' against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                        
                        if comparison_operator not in valid_operators:
                            print("Error: Given comparison operator is not in the list.")
                            invalid_operator = True
                            break

                        if feature_type == "str":
                            if comparison_operator != "==" and comparison_operator != "!=":
                                print("Error: Strings can only be compared using the equality operators '==' and '!='.")
                                invalid_operator = True
                                break

                        l_comparison_operators.append(comparison_operator)
                    elif target_feature in self.class_list:
                        class_index = self.class_list.index(target_feature)
                        class_type = self.class_types[class_index]

                        comparison_operator = input(f"Type in the comparison operator to compare the column '{target_feature}' against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                        
                        if comparison_operator not in valid_operators:
                            print("Error: Given comparison operator is not in the list.")
                            invalid_operator = True
                            break

                        if class_type == "str":
                            if comparison_operator != "==" and comparison_operator != "!=":
                                print("Error: Strings can only be compared using the equality operators '==' and '!='.")
                                invalid_operator = True
                                break

                        l_comparison_operators.append(comparison_operator)
                    elif target_feature == "distance_builtin":
                        comparison_operator = input(f"Type in the comparison operator to compare the distance against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                        
                        if comparison_operator not in valid_operators:
                            print("Error: Given comparison operator is not in the list.")
                            invalid_operator = True
                            break

                        l_comparison_operators.append(comparison_operator)
                    else:
                        column_index = self.columns.index(target_feature)
                        column_type = self.column_types[column_index]
                    
                        comparison_operator = input(f"Type in the comparison operator to compare the column '{target_feature}' against the value '{l_comparison_targets[idx]}' (must be '==', '!=', '<', '<=', '>=' or '>'): ").strip()
                        
                        if comparison_operator not in valid_operators:
                            print("Error: Given comparison operator is not in the list.")
                            invalid_operator = True
                            break

                        if column_type == "str":
                            if comparison_operator != "==" and comparison_operator != "!=":
                                print("Error: Strings can only be compared using the equality operators '==' and '!='.")
                                invalid_operator = True
                                break

                        l_comparison_operators.append(comparison_operator)

                if invalid_operator:
                    continue

                target_features = l_target_features
                comparison_targets = l_comparison_targets
                comparison_operators = l_comparison_operators

                break

            filtered_list = list(filter(lambda entry: self.custom_filter(entry, target_features, comparison_targets, comparison_operators), filtered_list))

            print("Samples have been filtered.")

        return filtered_list

    def string_to_num(self, string):
        if string in self.nan_values.keys():
            return self.nan_values[string]
        else:
            raise(f"Error converting string to number. Feature string '{string}' not found in the dataset.")
        
    def num_to_string(self, num):
        for string in self.nan_values.keys():
            if self.nan_values[string] == num:
                return string
            
        raise(f"Error converting number to string. Number '{num}' was not found in the dataset.")
   
    @staticmethod
    def normalize_helper(prev_value, prev_min, prev_max, new_min, new_max):
        if prev_max == prev_min:
            return 0
        
        prev_value_range = prev_max - prev_min
        percentage_of_previous_range = (prev_value - prev_min) / prev_value_range
        new_value_range = new_max - new_min

        result = new_min + (new_value_range * percentage_of_previous_range)
        if result < new_min:
            result = new_min
        elif result > new_max:
            result = new_max

        return result

    # Puts feature in [-1, 1] range (does not mean and variance normalize)
    def normalize(self, entry):
        # Ignores last feature (the dependent variable y to be predicted)
        for idx, feature in enumerate(self.feature_list):
            if entry == self.filtered_entries[0]:
                print(self.feature_min_maxes)
            normalized_value = Dataset.normalize_helper(
                entry.features[feature],
                self.feature_min_maxes[idx][0],
                self.feature_min_maxes[idx][1],
                -1,
                1
                )
            
            if self.feature_types[idx] == "str":
                for key in self.nan_values.keys():
                    if self.nan_values[key] == entry.features[feature]:
                        self.nan_values[key] = normalized_value
            
            entry.features[feature] = normalized_value
        
    # If `use_internal_entries` is `False`, `samples` and `samples_dependent_values` must be `np.matrix` variables of matching column and row dimensions, respectively
    # If `use_internal_entries` is `True`, `samples` and `samples_dependent_values` must be `None` (or simply ignored)
    # def get_dataset_stats(dataset=None, use_internal_entries=False, samples=None, samples_dependent_values=None, feature_list=None, class_list=None):
    @staticmethod
    def get_dataset_stats(samples=None, samples_dependent_values=None, feature_list=None, class_list=None):
        '''
        if use_internal_entries == False and (type(samples) != np.matrix or type(samples_dependent_values) != np.matrix):
            print(f"Error: Could not calculate dataset mean and variance, `use_internal_entries` was given as `False`, but `samples` and `samples_dependent_values` were not given (`np.matrix`).")
            exit(-1)
        elif use_internal_entries == True and (samples != None or samples_dependent_values != None):
            print(f"Error: Could not calculate dataset mean and variance, `use_internal_entries` was given as `True`, but `samples` and `samples_dependent_values` were not `None`/ignored.")
            exit(-1)
        '''

        '''
        if type(dataset) == None:
            features_mean = [0 for _ in range(len(feature_list))]
            features_variance = [0 for _ in range(len(feature_list))]
            features_std = [0 for _ in range(len(feature_list))]
            dependent_variables_mean = [0 for _ in range(len(class_list))]
            dependent_variables_variance = [0 for _ in range(len(class_list))]
            dependent_variables_std = [0 for _ in range(len(class_list))]
        elif type(dataset) == Dataset:
            features_mean = [0 for _ in range(len(dataset.feature_list))]
            features_variance = [0 for _ in range(len(dataset.feature_list))]
            features_std = [0 for _ in range(len(dataset.feature_list))]
            dependent_variables_mean = [0 for _ in range(len(dataset.class_list))]
            dependent_variables_variance = [0 for _ in range(len(dataset.class_list))]
            dependent_variables_std = [0 for _ in range(len(dataset.class_list))]
        else:
            print(f"Error: Could not get dataset stats, `dataset` argument was not of type `None` or `Dataset`. Its type is `{type(dataset)}`")
            exit(-1)
        '''

        if (type(samples) != np.matrix or type(samples_dependent_values) != np.matrix):
            print(f"Error: Could not calculate dataset mean, variance, and std, `samples` and `samples_dependent_values` were not given (as `np.matrix`).")
            exit(-1)

        features_mean = [0 for _ in range(len(feature_list))]
        features_variance = [0 for _ in range(len(feature_list))]
        features_std = [0 for _ in range(len(feature_list))]
        dependent_variables_mean = [0 for _ in range(len(class_list))]
        dependent_variables_variance = [0 for _ in range(len(class_list))]
        dependent_variables_std = [0 for _ in range(len(class_list))]
    
        # if not use_internal_entries:
        if samples.shape[1] <= 0 or samples_dependent_values.shape[1] <= 0:
            print(f"Error: Could not calculate dataset mean and variance, number of entries must be a positive integer. Was {samples.shape[1]} for the features/input vectors and {samples_dependent_values.shape[1]} for the dependent/output vectors.")
            exit(-1)

        if samples.shape[1] != samples_dependent_values.shape[0]:
            print(f"Error: Could not calculate dataset mean and variance, input and output vectors have mismatching dimensions. ({samples.shape[1]} vs {samples_dependent_values.shape[0]}).")
            exit(-1)

        num_entries = samples.shape[1]

        for sample in range(num_entries):
            for feature_idx, feature in enumerate(feature_list):
                features_mean[feature_idx] += samples[feature_idx, sample] / num_entries

            for class_idx, class_name in enumerate(class_list):
                dependent_variables_mean[class_idx] += samples_dependent_values[sample, class_idx] / num_entries

        for sample in range(num_entries):
            for feature_idx, feature in enumerate(feature_list):
                features_variance[feature_idx] += math.pow((samples[feature_idx, sample] - features_mean[feature_idx]), 2) / (num_entries - 1)

            for class_idx, class_name in enumerate(class_list):
                dependent_variables_variance[class_idx] += math.pow((samples_dependent_values[sample, class_idx] - dependent_variables_mean[class_idx]), 2) / (num_entries - 1)


        '''
        elif use_internal_entries:
            num_entries = len(dataset.filtered_entries)

            if num_entries <= 0:
                print(f"Error: Could not calculate dataset mean and variance, number of entries must be a positive integer, but was {num_entries}.")
                exit(-1)

            for entry in dataset.filtered_entries:
                for feature_idx, feature in enumerate(dataset.feature_list):
                    features_mean[feature_idx] += entry.features[feature] / num_entries

                for class_idx, class_name in enumerate(dataset.class_list):
                    dependent_variables_mean[class_idx] += entry.dependent_values[class_name] / num_entries

            for entry in dataset.filtered_entries:
                for feature_idx, feature in enumerate(dataset.feature_list):
                    features_variance[feature_idx] += math.pow((entry.features[feature] - features_mean[feature_idx]), 2) / num_entries

                for class_idx, class_name in enumerate(dataset.class_list):
                    dependent_variables_variance[class_idx] += math.pow((entry.dependent_values[class_name] - dependent_variables_mean[class_idx]), 2) / num_entries
        '''

        for feature_idx in range(len(feature_list)):
            features_std[feature_idx] = math.sqrt(features_variance[feature_idx])
        
        for class_idx in range(len(class_list)):
            dependent_variables_std[class_idx] = math.sqrt(dependent_variables_variance[class_idx])

        result = {}
        result["features_mean"] = features_mean
        result["features_variance"] = features_variance
        result["features_std"] = features_std
        result["dependent_variables_mean"] = dependent_variables_mean
        result["dependent_variables_variance"] = dependent_variables_variance
        result["dependent_variables_std"] = dependent_variables_std

        return result
                

class Entry:
    def __init__(self, dataset, feature_values, dependent_values, column_value_dict):
        self.features = {}
        self.dependent_values = {}

        for feature_name, feature_value in zip(dataset.feature_list, feature_values):
            self.features[feature_name] = feature_value

        for class_name, class_value in zip(dataset.class_list, dependent_values):
            self.dependent_values[class_name] = class_value

        self.column_value_dict = column_value_dict