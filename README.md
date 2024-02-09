# Used Cars In Saudi Arabia
### Collaborators
Jackson Rini, Sultan Alshakrah, Abdulrahman Baflah, Majed Zamzami
## Abstract
In this paper, we aim to use regression analysis to determine which used car brand maintains its value optimally based on the amount driven. We have discussed the need for this topic as well as examined similar studies that have been conducted related to our work. Our team has found two databases and combined them into one to have more data points to work on. Then, we also cleaned the data by getting rid of outliers, null values, and other values that hindered our work. We have created predictive models in three different methodologies: random forest regression, linear regression, and gradient boosting regression (XGBoost) to accomplish our goal. We evaluated the three models and by comparing the mean squared error and other factors we could choose which model to continue analysis on.
## Introduction
There has been a massive decline in the purchase of new cars in Saudi Arabia for many reasons. There has been a 60% decline in the sales of new cars since 2020 mainly because of the COVID-19 lockdown ( Impact of COVID-19 on the passenger car aftermarket in Saudi Arabia, 2020). In response to the financial and economic repercussions of COVID-19, Saudi Arabia implemented a substantial increase in its Value Added Tax (VAT) rate, raising it from 5% to 15% as of July 1, 2020. This VAT hike inevitably led to higher prices for new automobiles making them less appealing than before. New cars have become more and more out of reach of some people in Saudi Arabia. While the COVID-19 pandemic has significantly contributed to the boom in the used car market, other financial and economic factors have also played a pivotal role in expediting this growth. “Customers have postponed the purchase of a new car until there is more clarity in the near future. As a result, close to 50% of the buyers have expressed a strong desire to purchase high-quality used cars at a reasonable price” (Saudi Arabia’s used cars sales market is poised to take off, 2022). This makes the discussion of used cars and the depreciation of the price of a car more important than ever. In addition, used car dealers generally believe that Saudis tend to replace their vehicles every 2 to 4 years. This means we will probably see a huge shift in the ratio of used cars (20% in 2019) and new cars in favor of used cars. The factors for when to buy a used car or when to sell it have drastically changed. The usage of cars in Saudi Arabia in general is only going up. “GCG estimates that almost 75% of commuters in Saudi Arabia travel nearly 2 hours or more per day; those in Riyadh travel more than 2 hours”( Impact of COVID-19 on the passenger car aftermarket in Saudi Arabia, 2020). Furthermore, the increasing presence of women drivers on the roads has led to a greater demand for private driving licenses, a surge in motor insurance, the expansion of driving schools, and a notable uptick in car sales and leasing, ultimately benefiting the aftermarket sector.

This has caused the discussion of what used cars someone should buy for everyday use. Many points are brought up whether it is fuel consumption, country of manufacturing of the car, and the efficiency of the car in general. In order to understand the efficiency of cars and their brands we decided to understand what factors cause the car to depreciate and at what rate. The main objective of this project is to understand which cars hold their prices the most. To achieve this we needed to first be able to predict car prices in general. Then understand the change in car price based on some factors. This information could significantly impact the decision-making of the average Saudi citizen when purchasing a car, and allow people to make more informed decisions.
## Related Work
Mammadov (2021), a researcher at Carlo Bo University of Urbino, Italy, analyzed a dataset of used cars in the USA to make car predictions. He did this by using linear regression. His dataset had 25 attributes and over 100 thousand objects; much more than ours. He found correlations and built the linear regression based on that. He concluded that low-priced cars often have high mileage on them. 

Erdem, Associate Professor, and Senturk, Research Assistant, (2009) have developed a hedonistic regression on used cars in Turkey to determine what are the characteristics that affect the car price. One conclusion that they arrived at was that the model year of a car has a 5 percent significance level. This is helpful because one of the attributes in our dataset is the model year. Another one that is not useful is that the vehicles with the color black or gray tend to have a higher average price than other cars. The last one we would mention is the city factor. Different cities have different prices for the same car, Istanbul, the most populated city in Turkey, has a lower price compared to other cities in Turkey.

##Main Technique
###Data understanding:
We used Kaggle as the source for all our data. At first, we found a dataset  “Used Cars In Saudi Arabia” which has been web-scraped using BeautifulSoup from the Yallamotor website that sells used cars. This dataset includes only 2287 data points which quickly became a concern when looking at specific brands with a minimal amount of data. It also showed some information that was very unrealistic because of the low amount of data such as that the average price of an Audi was 1,119,380.00 SAR. This is why we decided to combine this dataset with another called “Saudi Arabia Used Car” which contains data that has been scraped from syarah.com. This data set contained 5,624 data points which gave us more to work with. This gave us a final larger dataset with attributes such as car brand, car model, car driven, car model year, and car price. To further understand the data we looked at the count data points for each brand and the average prices of each brand. This was conducted using Python, leveraging libraries like Pandas and Numpy for data manipulation and Seaborn for initial data visualization. In addition, we made a correlation heatmap to get an idea of the correlation between attributes. This gave us more information on what our following steps should be and how to proceed. In addition, insight into how to go about data cleaning and processing and what to expect from the results.

![Screenshot 2024-02-09 160704](https://github.com/jaxri/DataMiningProject/assets/64553469/1bb215dc-4de6-4974-b463-9e2d394bc54e)
	Fig 1. correlation matrix visualization

After this, we started analyzing data points that skew the data or give incorrect results. We did this by finding duplicates or cars being sold for free. In addition to this, we created distribution plots and spreads of price, Kilometers driven, and car model year. This allowed us to understand the ranges we wanted to work in and find outliers that were skewing the data as shown in Figure 2.

![Screenshot 2024-02-09 160937](https://github.com/jaxri/DataMiningProject/assets/64553469/32cfae46-27ad-4cda-8707-b020cb2d2486)
Fig 2. Price Distribution Plot and Price Spread

####Data Cleaning and Preprocessing:

Based on what we learned previously we deleted any data points with a price of less than 5000 SAR and more than 200000 SAR which fixed the Price distribution and Price spread significantly. We also did this with Kilometers driven, and car model year where we deleted data points with over 500000 kilometers driven and car model years under the year 2000. Even though this does not completely remove outliers it reduces them by a large amount. We then encoded categorical variables to prepare it for the model and make it machine-readable. Get_dummies was used to do this.  After that, we split the data into features (X) and target (y) and then into training and testing sets. This prepares the data for regression analysis.

####Methods and Algorithms:

In our project, we employed various data mining techniques to analyze and predict used car prices in Saudi Arabia. Our process began with data preprocessing, where we combined, cleaned, and refined our dataset. We implemented Three regression models, each with its unique strengths and suitability for the problem at hand. We utilized Scikit-learn for Random Forest, Linear Regression, and XGBoost for the Gradient Boosting model:

####Random Forest Regression: 

Random Forest Regression is a powerful ensemble technique known for its accuracy and ability to handle non-linear data.  This model is well-suited for handling the large feature space and complex relationships within our data. It is robust to overfitting and provides insights into feature importance. To implement this we generated a baseline using DummyRegressor. Then, we predicted using the baseline model and evaluated the baseline model. After that, we did Hyperparameter Optimization by using Optuna to optimize the Random Forest to find the best model for predictions. Metrics like Mean Absolute Error, Mean Squared Error, and R-squared were calculated using Scikit-learn's built-in functions. This resulted in an accuracy of 79.13% and prices that were on average off by 8925 SAR. 








####Linear Regression:

Linear Regression is a fundamental approach for predicting a quantitative response and understanding relationships between variables. It serves as a baseline for comparison. It is useful for understanding the direct linear relationships between the features and the target variable. To implement this we first create a linear regression model and split the data into training and testing sets. After this, we trained the linear regression model and used the trained model to make predictions on the test set. Lastly, we evaluate the model and get the Mean Squared Error, and R-squared using Scikit-learn's built-in functions. This model resulted in extremely inaccurate predictions as the mean squared error is very high. This is assumed since linear regression is sensitive to outliers, and assumes homoscedasticity, meaning the variance of error terms is constant across all levels of the independent variables.



####Gradient Boosting Regression (XGBoost):

Gradient Boosting Regression is an advanced ensemble technique that uses gradient boosting frameworks, known for its high performance and speed. It is, similar to random forest regression, well-suited for handling the large feature space and complex relationships within our data. To implement this we generated a baseline using DummyRegressor, evaluated the baseline model, used hyperparameter optimization by using Optuna, trained the model with the best hyperparameters, and then evaluated the best model. This resulted in an accuracy of 84.58% and prices that were on average off by 78235 SAR. Which makes it the most accurate model of the three. 






These models collectively provide a comprehensive analysis, each contributing unique insights, thereby enabling a thorough understanding of the factors influencing used car prices. Each step was carefully implemented and cross-validated to ensure the correctness of our approach. The use of Python and its associated libraries provided a robust and flexible environment for conducting comprehensive data analysis and model development.

##Evaluation

To obtain the wanted results of what car brands are most efficient in holding their price we need to get the depreciation rates of each brand. To set up the models for the evaluation, we first calculate an “actual depreciation rate” which is the car price divided by the amount of kilometers driven by the car. We then drop rows with NaN or infinite values and select the relevant features for our model. After that, we encode the categorical variables, split the data into training and testing sets, and train both the random forest model and the XGBoost model. Lastly, we evaluate the model, compare model predictions with actual depreciation rates, remove outliers, and visualize the results with a line of best fit. The line of best fit indicates the overall direction of the data points' relationship as shown in figure 3. Both lines have a positive slope, indicating a positive correlation between actual and predicted depreciation rates. As the actual rate increases, the predicted rate also increases. We see that the random forest model is better at predicting the depreciation rate despite it being less accurate in predicting the price in general.      



Fig 3. Visualization of the results with a line of best-fit


After this, we needed to obtain the average depreciation rate for all car brands to get a complete idea of which car brands hold their price the most and which depreciate a lot. We do this by extracting encoded 'car_brand' columns from the original dataset for 'results_rf',  grouping by car brand, and calculating the average depreciation rate. We then can use this to get a complete bar plot of the average depreciation rate for all car brands. By looking at the graph in Figure 4 and 5 we can see that some brands such as GMC, Mitsubishi, and Chevrolet depreciate very low amounts which makes them efficient in holding their price. We also see in figure 6 that Dodge and Chrysler fall in price the most, making them not very efficient cars from a financial standpoint.  When looking at the top 5 cars with the most depreciation we see all of them are brands from the United States which means that buying these cars in Saudi Arabia might not always be financially efficient. In addition, Hummer has an extremely high negative depreciation which means it depreciates the most and it has been removed from the graph to make the visualization clearer. This could be because maintenance of cars from the United States in Saudi Arabia is not as established as cars from Asia or Europe. Despite this, GMC and Chevrolet are still the cars that depreciate the least which might be because of the very high quality of these brands and their longevity, making them very good options for used car buyers. After examining the market sales in websites, we believe that while our results may not be entirely accurate, they still provide a valuable perspective. Some brands show positive depreciation which might be because of outliers, rare cars, and extremely low amounts of data since we can assume most normal cars should depreciate when driven more.   

Fig 4. Average depreciation rate XGBoost

Fig 5. Top 5 brands with lowest depreciation rate RF (Brands with above 50 cars)


Fig 6. Top 5 cars with most depreciation (Brands with above 50 cars)




Fig 7. Average depreciation rate RF

Conclusion

	Our study looked at which used cars in Saudi Arabia keep their value the best. We did a lot of research, gathered a bunch of data, cleaned it up, and then analyzed it using three different methods: random forest, linear regression, and XGBoost. After careful testing, we found out how fast different car brands lose value over time. Our discussion of the related work as well as the gathering, combing, cleaning, and preprocessing of the data led us to a better approach and method of analysis; which is running the three regression models and evaluating them until we choose the best model to work with. Then, we were able to obtain more accurate average depreciation rates for each brand and then evaluated the results and modified them further to get rid of the rare outliers and further improve the accuracy of the results.

In the end of the study, we have determined what car brands hold the most value which are GMC, Chevrolet, and Mitsubishi. These cars don't lose their value as quickly as other cars we examined. These findings can assist individuals looking to buy a used car in Saudi Arabia. They now have more knowledge on the matter and can make more informed decisions with strong confidence. Their money is better spent. Overall, this project gives people useful information about the used car market in Saudi Arabia and paves the way for more detailed studies later on.

Future Work

	Our project has been very limited due to the size of our data. The dataset had around 7000 objects at the start and was reduced after the cleaning. Hence, for some of the car brands, there was not enough data to make very precise predictions. Therefore, to improve the study, we would like to find more data, especially for the low data of car brands, and include newer or less popular car brands in Saudi Arabia to enhance the prediction and get more accurate depreciation rates. In the future, we plan to examine car data from other Middle Eastern countries that have markets similar to Saudi Arabia's. This comparison could help us understand how car prices fluctuate over time. Additionally, consulting with car experts and dealers could provide insights that aren't available in our current dataset. We also aim to investigate how broader economic factors, such as fuel prices, import taxes, and the 15% VAT in Saudi Arabia, influence car prices by affecting demand and supply. We understand that if demand exceeds supply, prices tend to increase, and conversely, if supply exceeds demand, prices usually decrease. These economic elements can significantly impact a car's value, offering us a fuller understanding of the entire Saudi car market.

Furthermore, the incorporation of different prediction analyses might get us a more accurate analysis. One method that was taken into consideration by my team was deep neural networks. We believe it would have provided more precise results with less mean variance thus concluding with more robust work.

References
- Pike, J. (n.d.). MK 15 Phalanx Close-In Weapons System (CIWS). Mk 15 phalanx close-in weapons system (CIWS). https://man.fas.org/dod-101/sys/ship/weaps/mk-15.htm
- Consultancy-me.com. (2022, October 18). Saudi Arabia’s used cars sales market is poised to take-off. Consultancy. https://www.consultancy-me.com/news/5464/saudi-arabias-used-cars-sales-market-is-poised-to-take-off#:~:text=Saudi%20Arabia%27s%20used%20car%20and,of%20%2428.7%20billion%20by%202025
- plantmachineryvehicles.com (2020). Impact of COVID-19 on the passenger car aftermarket in Saudi Arabia. plantmachineryvehicles. https://www.plantmachineryvehicles.com/equipment/vehicles/77772-impact-of-covid-19-on-the-passenger-car-aftermarket-in-saudi-arabia
- Mammadov, H. (2021). CAR PRICE PREDICTION IN THE USA BY USING LINEAR REGRESSION. Journal of Economic Behavior, 11. 

- Hedonic analysis of used car prices in Turkey* - researchgate. (2009). https://www.researchgate.net/profile/Ismail-Sentuerk/publication/261831282_A_Hedonic_Analysis_of_Used_Car_Prices_in_Turkey/links/0deec53590f0e5045d000000/A-Hedonic-Analysis-of-Used-Car-Prices-in-Turkey.pdf
- alruqi, R. (2021, January 31). Used cars in Saudi Arabia. Kaggle. https://www.kaggle.com/datasets/reemalruqi/used-cars-in-saudi-arabia
- Muhith, R. (2023, February 22). Saudi Arabia Used Car. Kaggle. https://www.kaggle.com/datasets/raihanmuhith/saudi-arabia-used-car
- Muhith, R. (2023, February 22). Used Car Price Prediction. Kaggle. https://www.kaggle.com/code/raihanmuhith/used-car-price-prediction
- Khaiid. (2022, March 26). 🚗🚘 car price prediction [r2 ≈ 0.88]. Kaggle. https://www.kaggle.com/code/khaiid/car-price-prediction-r-0-8
