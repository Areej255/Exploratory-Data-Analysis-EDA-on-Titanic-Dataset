import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from a CSV file into a pandas DataFrame (table)
df = pd.read_csv('C:\\Users\\PMLS\\Downloads\\train.csv') 
print(df.head()) #this will print the first five rows 
print(df.isnull().sum())  # This shows how many missing values are in each column

df['Age'].fillna(df['Age'].median(), inplace=True)  # Fill missing values in 'Age' with the 
# median (middle) value
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Fill missing values in 'Embarked'
# with the most frequent value (mode)

df.drop(columns=['Cabin'], inplace=True)  # Drop the 'Cabin' column because it has too many missing values
df['Survived'] = df['Survived'].astype('category')  # Convert the 'Survived' column to a category for better analysis

print(df.dtypes)  # Show the types of each column in the dataset (e.g., integers, floats, categories)


# gender class rate
gender_survival = df.groupby('Sex')['Survived'].mean()  # Group the data by gender and calculate the survival rate
print(gender_survival)  # Print the result

sns.barplot(x='Sex', y='Survived', data=df)  # Create a bar plot showing survival rate by gender
plt.title('Survival Rate by Gender')  # Set the title for the plot
plt.show()  # Display the plot

# # passenger class rate
class_survival = df.groupby('Pclass')['Survived'].mean()  # Group the data by passenger class and calculate the survival rate
print(class_survival)  # Print the result

sns.barplot(x='Pclass', y='Survived', data=df)  # Create a bar plot showing survival rate by passenger class
plt.title('Survival Rate by Passenger Class')  # Set the title for the plot
plt.show()  # Display the plot

# age survival rate 
plt.figure(figsize=(10,6))  # Set the size of the figure (plot)
df['Age'].hist(bins=30, color='lightblue', alpha=0.7)  # Create a histogram to show the distribution of ages
plt.title('Age Distribution')  # Set the title for the histogram
plt.xlabel('Age')  # Label the X-axis
plt.ylabel('Frequency')  # Label the Y-axis
plt.show()  # Display the plot


df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 12, 18, 50, 80], labels=['Child', 'Teen', 'Adult', 'Senior'])  # Group passengers by age into categories
age_group_survival = df.groupby('AgeGroup')['Survived'].mean()  # Calculate survival rate by age group
print(age_group_survival)  # Print the result

sns.barplot(x='AgeGroup', y='Survived', data=df)  # Create a bar plot showing survival rate by age group
plt.title('Survival Rate by Age Group')  # Set the title for the plot
plt.show()  # Display the plot


# additional visualizations
survived_gender = df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack()  # Calculate the proportion of survivors by gender
survived_gender.plot(kind='pie', subplots=True, autopct='%1.1f%%', figsize=(12, 6))  # Create a pie chart
plt.title('Survival Distribution by Gender')  # Set the title for the pie chart
plt.ylabel('')  # Remove Y-axis label for a cleaner look
plt.show()  # Display the chart




