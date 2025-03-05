            
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Preprocessor:
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def preprocess(self, data=None):
        # Apply various preprocessing methods on the DataFrame
        
        if data is None:
            data = self.df
            
        self.df = self._preprocess_numerical(self.df)
#         self.df = self._preprocess_categorical(self.df)
#         self.df = self._preprocess_ordinal(self.df)
        return self.df

    def _preprocess_numerical(self, df):
    # Custom logic for preprocessing numerical features goes here

    # Create a copy of the DataFrame to avoid SettingWithCopy issues
        df = df.copy()

    # Detect and replace the outlier
        def cap_data(df):
            for col in df.columns:
                if (((df[col].dtype) == 'float64') | ((df[col].dtype) == 'int64')):
                    percentiles = df[col].quantile([0.01, 0.99]).values
                    
                    #value <= 1st percentile, round value to the 1st percentile
                    df[col][df[col] <= percentiles[0]] = round(percentiles[0], 6)
                    #value => 99st percentile, round value to the 99st percentile
                    df[col][df[col] >= percentiles[1]] = round(percentiles[1], 6)
                else:
                    #convert to num, if cannot, then Nan
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        cap_data(df)

    # Only for the numerical features
    # Standardization 
    # Replace the null values with mean
        for feature in df.columns[0:18]:
            df[feature].fillna(df[feature].mean(), inplace=True)
            df[feature] = (df[feature] - df[feature].mean()) / df[feature].std()

    # Only for the Binary features
    # Replace the null values with mode of column [18:77] for binary features
        for x in df.columns[18:77]:
            df[x].fillna(df[x].mode().iloc[0], inplace=True)

        return df



class NaiveBayesClassifier():
    def __init__(self):
        # Initialize the classifier
        self.classes = None
        self.prior = None
        self.num_means = None
        self.num_variances = None
        self.bin_probabilities = None

    def fit(self, X, y, alpha=1):
        # Separate numerical and binary features
        X_num = X.iloc[:, :17]
        X_bin = X.iloc[:, 17:]

        # get each unique classes and calculate the priors
        self.classes = np.unique(y)
        self.prior = {classes: (np.sum(y == classes) + alpha) / (len(y) + 
                                len(self.classes) * alpha) for classes in self.classes}

        # Calculate means and variances for numerical features with Laplace smoothing
        self.num_means = (X_num.groupby(y).sum() + alpha) / (X_num.groupby(y).count() + alpha * 17)
        self.num_variances = (X_num.groupby(y).apply(lambda x: ((x - x.mean()) ** 2).sum()) + alpha) / (X_num.groupby(y).count() + alpha * 17)

        # Calculate probabilities for binary features with Laplace smoothing P(x|0), P(x|1)
        self.bin_probabilities = (X_bin.groupby(y).sum() + alpha) / (X_bin.groupby(y).count() + alpha * (X_bin.shape[1]))
        
        #print(self.bin_probabilities)


    def predict(self, X):
        predictions = []
        
        #a loop to iterate through index and each rows
        for na, instance in X.iterrows():
            class_likelihoods = []

            for classes in self.classes:
                #get the logarithm of the prior probability for classes P(c)
                prior = np.log(self.prior[classes])
                #Gaussian probability density function for the numerical features [1:17]
                num_likelihood = np.sum(
                    -0.5 * np.log(2 * np.pi * self.num_variances.loc[classes])
                    - 0.5 * ((instance[:17] - self.num_means.loc[classes]) ** 2 / self.num_variances.loc[classes])
                )
                #Log probability for the binary features [17:77]
                bin_likelihood = np.sum(
                    instance[17:] * np.log(self.bin_probabilities.loc[classes]) +
                    (1 - instance[17:]) * np.log(1 - self.bin_probabilities.loc[classes])
                )

                # cuz of log, use + instead of x, the den can be ignored
                total_likelihood = prior + num_likelihood + bin_likelihood
                class_likelihoods.append(total_likelihood)
            
            #take the max class_likelihoods
            predicted_class = self.classes[np.argmax(class_likelihoods)]
            predictions.append(predicted_class)

        return predictions

    def predict_proba(self, X):
        probabilities = []

        for na, instance in X.iterrows():
            class_likelihoods = []

            for classes in self.classes:
                prior = np.log(self.prior[classes])
                num_likelihood = np.sum(
                    -0.5 * np.log(2 * np.pi * self.num_variances.loc[classes])
                    - 0.5 * ((instance[:17] - self.num_means.loc[classes]) ** 2 / self.num_variances.loc[classes])
                )
                bin_likelihood = np.sum(
                    instance[17:] * np.log(self.bin_probabilities.loc[classes]) +
                    (1 - instance[17:]) * np.log(1 - self.bin_probabilities.loc[classes])
                )

                total_likelihood = prior + num_likelihood + bin_likelihood
                class_likelihoods.append(total_likelihood)
            
            #softmax function: ensuring sum up to 1 for each instance
            exp_likelihoods = np.exp(class_likelihoods - np.max(class_likelihoods))
            class_probabilities = exp_likelihoods / np.sum(exp_likelihoods)
            probabilities.append(class_probabilities)

        return np.array(probabilities)
    
    
# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return (distance)**(1/2)

# K-Nearest Neighbors Classifier
class KNearestNeighbors:
    def __init__(self, k=7):
        # Initialize KNN with k neighbors
        self.k = k

    def fit(self, X, y):
        # Store training data and labels for KNN
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Implement the prediction logic for KNN
        predictions = [self._predict(x) for x in X.values]
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Implement probability estimation for KNN
        probabilities = []
        for x in X.values:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train.values]
            k_idx = np.argsort(distances)[:self.k]
            k_neighbor_labels = self.y_train.iloc[k_idx]

            # count for each class label
            class_counts = {}
            for label in k_neighbor_labels:
                #count the current label, if the label not yet in the dict, then retunr 0
                # +1 means increments the count by 1,
                class_counts[label] = class_counts.get(label, 0) + 1

            # Convert counts to probabilities
            class_probs = [class_counts.get(c, 0) for c in sorted(np.unique(self.y_train))]
            #print(class_probs)
            
            
            probabilities.append(class_probs)

        #normalized by dividing by self.k to ensure they sum up to 1 for each instance.
        return np.array(probabilities) / self.k

    def _predict(self, x):
        # Helper method for prediction
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train.values]
        #Find the indices of the k samples with the smallest distances.
        k_idx = np.argsort(distances)[:self.k]
        #locate the class labels with the indices with smallest distance 
        k_neighbor_labels = self.y_train.iloc[k_idx]
        
        #count each unique class label in k neighbours
        unique_labels, counts = np.unique(k_neighbor_labels, return_counts=True)
        #print(unique_labels)
        #return the highest amount of unqiue class label
        most_label = unique_labels[np.argmax(counts)]
        #print(most_label)

        return most_label



# Multilayer Perceptron Classifier
class MultilayerPerceptron():
    def __init__(self, hidden_layers_sizes = 10):
        # Initialize MLP with given network structure
        self.para = {}
        self.hidden_layers_sizes = hidden_layers_sizes
        self.loss = []
        self.X = None
        self.y = None
        self.input_size = None
        self.output_size = None
        
        pass
    
    def init_weights(self):
        
        np.random.seed(1)

        self.para["W1"] = np.random.randn(self.input_size, self.hidden_layers_sizes)
        self.para["b1"] = np.random.randn(self.hidden_layers_sizes,)
        self.para["W2"] = np.random.randn(self.hidden_layers_sizes, self.output_size)
        self.para["b2"] = np.random.randn(self.output_size,)
        
    # prevent division by zero
    def eps(self, x):
        epsilon = 0.00000000001
        return np.clip(x, epsilon, None)
    
    def sigmoid(self, Z):
        sig = 1/(1+np.exp(-Z))
        return sig
    
    def relu(self, Z):
        return np.maximum(0,Z)
        
    def dRelu(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x    
    
    def entropy_loss(self, y, output):
        #computes the cross-entropy loss between the predicted output and the actual labels y
     
        output_inv = 1.0 - output
        y = y.to_numpy().reshape(output.shape) 
        y_inv = 1.0 - y
        output = self.eps(output)
        output_inv = self.eps(output_inv)
        
        loss = -1/len(y) * np.sum(y * np.log(self.eps(output)) + (1 - y) * np.log(self.eps(1 - output_inv)))

        return loss
        
    def fit(self, X, y, learning_rate = 0.001, epochs = 150):
        # Implement training logic for MLP including forward and backward propagation
        
        self.X = X
        self.y = y
        self.input_size = X.shape[1]
        self.output_size = 1  
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.init_weights()
        
        for i in range(self.epochs):
            output, loss = self._forward_propagation(X)
    
            gra_w1, gra_w2, gra_b1, gra_b2 = self._backward_propagation(output)
            
            #update the weights and bias
            self.para['W1'] = self.para['W1'] - self.learning_rate * gra_w1
            self.para['W2'] = self.para['W2'] - self.learning_rate * gra_w2
            self.para['b1'] = self.para['b1'] - self.learning_rate * gra_b1
            self.para['b2'] = self.para['b2'] - self.learning_rate * gra_b2
            
            self.loss.append(loss)
            
        pass

    def predict(self, X):
        # Implement prediction logic for MLP
                        
        L1 = X.dot(self.para["W1"]) + self.para["b1"]
        A1 = self.relu(L1)
        L2 = A1.dot(self.para["W2"]) + self.para["b2"]
        
#         Z1, A1, Z2, na, na = self._forward_propagation(X)
        pred = self.sigmoid(L2)
        return np.round(pred)
        pass

    def predict_proba(self, X):
        # Implement probability estimation for MLP
        
        L1 = X.dot(self.para["W1"]) + self.para["b1"]
        A1 = self.relu(L1)
        L2 = A1.dot(self.para["W2"]) + self.para["b2"]
        output = self.sigmoid(L2)
        return output
        
#         Z1, A1, Z2, output, loss = self._forward_propagation(X)
#         return output

        pass
        
    def _forward_propagation(self,X):
        # Implement forward propagation for MLP
        
        L1 = self.X.dot(self.para["W1"]) + self.para["b1"]
        A1 = self.relu(L1)
        L2 = A1.dot(self.para["W2"]) + self.para["b2"]
        output = self.sigmoid(L2)
        loss = self.entropy_loss(self.y, output)
        
        self.para["L1"] = L1
        self.para["L2"] = L2
        self.para["A1"] = A1
        
        
        return output, loss
#         return Z1, A1, Z2, output, loss
        pass

    def _backward_propagation(self, output):
        # Implement backward propagation for MLP
        
        #Compute the inverse of the target labels 
        y_inv = 1-self.y.to_numpy().reshape(output.shape)
        output_inv = 1 - output
        
        #reshape y to match the output shape
        y_np = self.y.to_numpy().reshape(output.shape)
        
        #gradient of loss
    
        gra_output = (y_inv / self.eps(output_inv)) - (y_np / self.eps(output))

        # sigmoid activation in the second layer
        #the formula of derivative of sigmoid function
        gra_sig = output * (output_inv)
        #gradient to the weighted sum with second layer
        gra_l2 = gra_output * gra_sig

        # the activation of the first layer
        gra_A1 = gra_l2.dot(self.para["W2"].T)
        gra_w2 = self.para["A1"].T.dot(gra_l2)
        # Calculate the gradient of the biases in the second layer 
        # by summing the gradient in the second layer along the axis 0 (rows).
        gra_b2 = np.sum(gra_l2, axis=0)

        gra_l1 = gra_A1 * self.dRelu(self.para['L1'])
        gra_w1 = self.X.T.dot(gra_l1)
        
        # Calculate the gradient of the biases in the first layer 
        # by summing the gradient in the second layer along the axis 0 (rows).
        gra_b1 = np.sum(gra_l1, axis=0)
        
        return gra_w1, gra_w2, gra_b1, gra_b2
    
        pass
    
    

# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test)

        
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # Check if the model supports predict_proba method for AUC calculation
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        
        if isinstance(proba, pd.DataFrame):
            proba = proba.to_numpy()         

        if len(np.unique(y_test)) == 2:  # Binary classification
            if proba.ndim > 1 and proba.shape[1] > 1:
                # Ensure y_test is binary
                if len(np.unique(y_test)) > 2:
                    y_test = (y_test == positive_class_label).astype(int)  
                auc = roc_auc_score(y_test, proba[:, 1])
            else:
                auc = roc_auc_score(y_test, proba)
                
            #print(f"AUC: {auc}")

        else:  # Multiclass classification
            # Ensure y_test is not multilabel
            if len(np.unique(y_test)) > 2:
                y_test = (y_test == positive_class_label).astype(int)  
            auc = roc_auc_score(y_test, proba, multi_class='ovo')
            #print(f"AUC: {auc}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }

def manual_train_val_split(data, test_size=0.2, random_state=None):
    # Manual implementation of train-validation split
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
    
# Main function to execute the pipeline
def main():
    # Load trainWithLable data
    df = pd.read_csv('trainWithLabel.csv')
    
    # Manually split your data into training and validation sets
    X_train, X_val = manual_train_val_split(df.drop('Outcome', axis=1), test_size=0.2, random_state=42)
    y_train, y_val = df['Outcome'].iloc[X_train.index], df['Outcome'].iloc[X_val.index]
    
    # Preprocess the training data
    preprocessor = Preprocessor(X_train)
    X_train_processed = preprocessor.preprocess()
    
    X_val_processed = preprocessor.preprocess(X_val)

    # Define the models for classification
    models = {'Naive Bayes': NaiveBayesClassifier(), 
              'KNN': KNearestNeighbors(),
              'MLP': MultilayerPerceptron()
    }

    # Split the dataset into features and target variable
#     X_train = df_processed.drop('Outcome', axis=1)
#     y_train = df_processed['Outcome']
#     df_processed['Outcome'] = df_processed['Outcome'].astype('category')


    # Perform K-Fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = []

    for model_name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_processed), start=1):

            X_train_fold, X_val_fold = X_train_processed.iloc[train_index], X_val_processed.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
#             # Apply preprocessing only to the training set
#             X_train_fold_processed = preprocessor.preprocess(X_train_fold)
#             X_val_fold_processed = preprocessor.preprocess(X_val_fold)


            model.fit(X_train_fold, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold, y_val_fold)
            fold_result['model'] = model_name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)


    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]

    print("Cross-validation results:")
    print(all_results_df)

    # Save results to an Excel file
    all_results_df.to_excel('cv_results.xlsx', index=False)
    print("Cross-validation results with averages saved to cv_results.xlsx")

    # Load the test dataset, assuming you have a test set CSV file without labels
    df_ = pd.read_csv('testWithoutLabel.csv')
    preprocessor_ = Preprocessor(df_)
    X_test = preprocessor_.preprocess()

    # Initialize an empty list to store the predictions of each model
    predictions = []

    # Make predictions with each model
    for name, model in models.items():        
        model_predictions = model.predict(X_test)
        predictions.append({
            'model': name,
            'predictions': model_predictions
        })

    # Convert the list of predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predictions
    print("Model predictions:")
    print(predictions_df)

    # Save the predictions to an Excel file
    predictions_df.to_csv('test_results.csv', index=False)
    print("Model predictions saved to test_results.xlsx")

if __name__ == "__main__":
    
    main()
