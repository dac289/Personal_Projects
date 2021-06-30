'''
Created on 06/29/2021

This program is for a Cheese Pricing Case Study

The objective was to sell more cheese, with losing as little margin as possible

Author: David Cresap
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVR
from statistics import mean


def get_data(filename, headers):
    '''
    Pull in the data on cheese, add the column names, 
    and remove any NA values from the data
    '''

    df = pd.read_excel(filename, engine='openpyxl',skiprows=2)
    pd.set_option('display.max_columns',30)
    df.columns = headers
    df = df.dropna()
    return df

def data_for_ml(df):
    '''Add two columns for price and demand changes as percentages, 
    this data will be used to build the SVR machine learning algorithm 
    so it can be applied to all values in the dataset'''

    df['percent price change'] = df['PRICE (JUNE)']/df['PRICE (MAY)']
    df['percent unit change'] = df['UNITS (JUNE)']/df['UNITS (MAY)']
    return df

def mach_learn(df):
    """
    The 'percent price change' and 'percent unit change' columns 
    will be used in the SVR model

    A test percent price change dataset is created to have 0.01 set 
    incraments from the lowest to hight price change, this new dataset
    will be used to show a smooth affect price has on demand at different 
    percent price changes

    A graph is created at the end to visually show how the test price data
    affects demand
    """

    # Change values to Numpy array
    X = df[['percent price change']].values
    y = df[['percent unit change']].values

    # Creating the test dataset using the min and max values and
    # using 0.01 as the step value from min to max values
    x_max = np.round(max(X),2)
    x_min = np.round(min(X),2)
    diff_x = np.arange(x_min, x_max, 0.01)
    diff_x = diff_x.reshape((len(diff_x),1))

    # Scaling the data to be used in the SVR machine learning model
    scaler_x = StandardScaler()
    scaler_x.fit(X)
    X_train = scaler_x.transform(X)
    X_test = scaler_x.transform(diff_x)
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y)

    # Using the SVR model 'rbf' to create a smooth non-linear model, 
    # becuase price/demand curves are not linear
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    y_pred_fix = scaler_y.inverse_transform(y_pred) # Change the y values back into percentages

    # Graph of how percent price change affects percent demand
    plt.figure(figsize=(8,6))
    plt.style.use('seaborn')
    plt.plot(diff_x,y_pred_fix)
    plt.title('Breakdown of Model, Price vs Demand',fontsize=20)
    plt.xlabel('Price Change',fontsize=14)
    plt.ylabel('Demand Change',fontsize=14)
    plt.show()

    # Return the Test dataset and predicted values to be used on the dataframe later
    return diff_x,y_pred_fix

def find_new_price_values(df,x_test, y_pred):
    """
    This applies the the test percent price change and predicted demand percent change 
    to each price and demand value in June, because it is the most recent data

    Because I did not know if the cost of each unit was going to increase or decrease
    I averaged the cost from May and June to account for either possibility

    The test percent price change was applied to the June Price for each item
    and the predicted demand change was applied to each June Units Sold

    Then calculated Total sales (Price * Demand) and the 
    price over the competition (Price - Competition Price)

    Then created a dataframe that had all 4 elements in it

    Using StandardScaler to scale down Price, Demand, Total Sales to have each 
    value have equal effect when added together to find the optimal balance between 
    all four values

    StandardScaler was not applied to price over competition, becasue I wanted 
    the price of the items over the competition to have a good effect on the outcome
    and usually it was small enought to not have a major effect on optimal balance

    I then found the index of the max value when all columns were added together
    and appended the "new" price and demand to the new_values list

    Last, created a dataframe from the new_values list and returned it
    """

    new_values = []
    for p in range(len(df)):
        """
        This loop goes through each price and demand value in June and applies the 
        test price percent change and predicted demand percent change
        """
        startprice = round(df['PRICE (JUNE)'][p], 2) # Using June as starting price
        cost = (df['UNIT COST (JUNE)'][p] + df['UNIT COST (MAY)'][p]) / 2 # Averaging cost from May and June
        demand_start = df['UNITS (JUNE)'][p] # Using June's demand as starting demand

        NewPrice = [np.round(startprice * x, 2) for x in x_test] # Applying the price percentage change to startprice
        NewDemand = [np.round(demand_start * x, 2) for x in y_pred] # Applying demand percentage change to demand_start

        # Calculating Total Sales values by multiplying the NewPrice * NewDemand
        NewTotalSales = [np.round((NewPrice[x]) * NewDemand[x], 2) for x in range((len(NewPrice) - int(startprice)))]
        # Calculating Price over competition
        PriceOverComp = [np.round(NewPrice[x]-df['COMPETITOR PRICE (JUNE)'][p],2) for x in range(len(NewPrice))]
        
        # Creating new dataframe
        tableOfPriceInfo = pd.DataFrame(data=[NewPrice,NewDemand,NewTotalSales,PriceOverComp]).transpose()

        # Using StandardScaler to make price, demand, and total sales equal in effect on optimal outcome
        sc1 = StandardScaler()
        sc2 = StandardScaler()
        sc3 = StandardScaler()
        
        sc1.fit(tableOfPriceInfo[[0]].values)
        sc2.fit(tableOfPriceInfo[[1]].values)
        sc3.fit(tableOfPriceInfo[[2]].values)

        tableOfPriceInfo[0] = sc1.transform(tableOfPriceInfo[[0]].values)
        tableOfPriceInfo[1] = sc2.transform(tableOfPriceInfo[[1]].values)
        tableOfPriceInfo[2] = sc3.transform(tableOfPriceInfo[[2]].values)

        tableOfPriceInfo.dropna(inplace=True) # Dropping NA values that might mess up optimal values

        # Summing columns together to create optimal values
        tableOfPriceInfo['Optimized'] = tableOfPriceInfo.sum(axis=1)
        
        max_optim_index = tableOfPriceInfo['Optimized'].idxmax() # Finding max optimal value index

        # Reversing Standard Scaler
        tableOfPriceInfo[0] = sc1.inverse_transform(tableOfPriceInfo[[0]].values)
        tableOfPriceInfo[1] = sc2.inverse_transform(tableOfPriceInfo[[1]].values)
        
        # Appending price and demand values at the max optimal index to new_values list
        new_values.append([tableOfPriceInfo[0][max_optim_index],tableOfPriceInfo[1][max_optim_index]])

    df_addon = pd.DataFrame(data=new_values) # Creating dataframe of optimal values for each cheese
    print(df_addon)
    
    return df_addon

def calculations_of_predicted_vs_observed(df):
    """
    Calculating original objective of selling more cheeses with losing as little margin as possible

    First: Calculating predicted values for year (NewPrice * NewDemand) * 12, and comparing it to
    52 week value from original dataframe

    Second: Calculating profit loss

    Third: Calculating increase of Cheeses sold
    """

    # Calculating new total sales
    newTotal_52weeks = sum(df['NewPrice'] * df['NewDemand']) * 12
    oldTotal_52weeks = sum(df['SALES (52 WEEKS)'])
    # Comparing total sales
    difference_of_52weeks = round(newTotal_52weeks - oldTotal_52weeks,2)
    print(difference_of_52weeks)

    # Calculating Profit
    newProfit = sum((df['NewPrice'] - ((df['UNIT COST (MAY)'] + df['UNIT COST (JUNE)'])/2))* df['NewDemand'])
    juneProfit = sum((df['PRICE (JUNE)'] - df['UNIT COST (JUNE)']) * df['UNITS (JUNE)'])
    # Comparing profits
    profit_month_comparison = round(newProfit - juneProfit,2)
    print(profit_month_comparison)

    # Calculating Demand Total
    newTotalDemand = sum(df['NewDemand'])
    avgOldDemand = sum((df['UNITS (MAY)'] + df['UNITS (JUNE)'])/2)
    # Comparing Demand Total
    demand_month_comparison = int(newTotalDemand - avgOldDemand)
    print(demand_month_comparison)


def main():
    filename = "Commodity Cheese Data - Pricing Analyst Interview Case Study.xlsx"
    headers = ['CATEGORY','SUBCATEGORY','BRAND','PRICING LINE','UPC','PRODUCT DESCRIPTION','PACK SIZE','SIZE',
               'UOM', 'PRICE (MAY)','UNIT COST (MAY)','UNITS (MAY)','SALES (MAY)','COMPETITOR PRICE (MAY)','PRICE (JUNE)','UNIT COST (JUNE)','UNITS (JUNE)',
               'SALES (JUNE)','COMPETITOR PRICE (JUNE)', 'SALES (52 WEEKS)']
    df = get_data(filename,headers)
    df = data_for_ml(df)
    x_test, y_pred = mach_learn(df)
    df_addon = find_new_price_values(df, x_test, y_pred)
    df_addon.columns = ['NewPrice', 'NewDemand'] # Add column names to new created dataframe

    # Concat the New dataframe with New prices and demands to original dataframe
    df = pd.concat([df,df_addon], axis=1) 
    print(df)

    calculations_of_predicted_vs_observed(df)

if __name__ == "__main__":
    main()

