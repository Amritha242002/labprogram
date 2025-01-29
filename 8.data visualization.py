import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('sales.csv')
def display_plots(choice):
    if choice == '1':  
        plt.scatter(data['Item Type'],data['Unit Price'])
        plt.title('Scatter Plot ')
        plt.xlabel('Unit Price')
        plt.ylabel('Item ')
        plt.xticks(rotation=45, ha='right') 
      
    elif choice == '2':
        counts,bins,patches=plt.hist(data['Unit Price'], bins=20, color='green', edgecolor='black')
        plt.title('Histogram of Unit Price')
        plt.xlabel('Unit Price')
        plt.ylabel('Frequency')
        plt.xticks(bins,rotation=45)
        

    elif choice == '3':
        mycolors=["green","blue","pink","yellow"]
        tips = sns.load_dataset('tips') 
        tips.head() 
        sns.set_style("whitegrid") 
        sns.boxplot(x = 'day', y = 'total_bill', data = tips,palette=mycolors) 
        plt.ylabel("Total Bill")
        plt.xlabel("Day")
        plt.show()

    elif choice == '4':
        plt.bar(data['Item Type'],data['Unit Price'])
        plt.title('Bar Chart of  Unit Price by Item Type')
        plt.xlabel('Item Type')
        plt.ylabel('Unit Price')
        plt.xticks(rotation=45, ha='right') 

    elif choice == '5':  
        item_counts = data['Item Type'].value_counts()  
        item_counts.plot(kind='pie', autopct='%1.1f%%', colors=plt.cm.Paired.colors)
        plt.title('Pie Chart of Item Type')
        plt.ylabel('')

    plt.show()
while True:
    print("\nChoose an option:")
    print("1. Show Scatter Plot")
    print("2. Show Histogram")
    print("3. Show Box Plot")
    print("4. Show Bar Chart")
    print("5. Show Pie Chart")
    print("6. Exit")
    
    choice = input("Enter your choice: ")

    if choice == '6':
        print("Exiting the program.")
        break  

    if choice in ['1', '2', '3', '4', '5']:
        display_plots(choice)
    else:
        print("Invalid choice.")
