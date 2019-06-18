# kaggle.Titanic

This kaggle competition is concerned with predicting deaths on Titanic. Even though the list of people who passed on the infamous ship is publicly available, the challenge remains an interesting place to start tinkering with machine learning.

## Analysis

The set comprises of 12 columns - 11 features and one target vector - the `Survived` column.

1. `PassengerId` is a unique identifier for each passenger. The column can be used to compare our results at the end, however can be droped in the training phase, as it adds no real value to the algorithm.
2. `Pclass` is the passenger class - a categorical variable with 3 classes. This seems to be an important column, as higher-class passengers should typically have higher priority to be saved. Indeed, as the figure below shows, the survival rates for 1st class passengers were significantly higher.  
   ![alt text](img/pclass_rel.png)
3. `Name` is the passenger class - a categorical variable with 3 classes. Upon inspection, it can be discovered that this column contains also the passenger's title - _Mr._/_Mrs._/_Miss._ etc.. It is possible to extract multiple classes for the various titles of the people on board. It could be expected that people with rare titles (e.g. _Countess_, _Don_, etc.) will have higher priority. Despite that, the same people will also likely have a higher passenger class. Additionally, we can determine the gender of the passenger, if that information was missing, but that is not the case in either of the data sets.
4. `Sex` is the gender of the passenger - categorical variable which can be either _male_ or _female_. It is a well known fact that women and children on Titanic were more likely to be saved, hence this column should have a high significance as a predictor. Indeed, if we look at the survival rate, a high percentage of females survived, contrary to the males column
   ![alt text](img/gender_rel.png)
5. `Age` is the age of the passenger - a continuous variable. As also mentioned in the previous section, age should have a high significance for the predictor. As it can be difficult to visualise and work with continuous variables, we can choose to categorize it, splitting the age into multiple ranges. Five bins were chosen in the example below, where we can see that passengers younger than 16 have had a much higher survival rate, contrary to elderly people above 64. Age is the first column where we observe missing entries in the dataset. Since the passenger age column has a normal distribution, a mean imputer would be a suitable choice to replace the missing values.
   ![alt text](img/age_rel.png)
6. `SibSp` defines the number of siblings the passenger had on board. It is most definitely a good predictor and should be used in the model. We can use it individually, or combine it with Parch, to get a more complete picture of the family survival rate on Titanic.
7. `Parch` defines the number of parents or children the passenger had on board. Together with the `SibSp` we can form a new column which can indicate the amount of family the passenger had with him on board. A nice observation from the illustration below is that people who were alone on board had a much lower survival rate.
   ![alt text](img/family_size_rel.png)
8. `Ticket` is a unique identifier for the ticket. While this information may be somewhat useful for filling out missing values in other features, it is highly unlikely that it will add any value to our model.
9. `Fare` is another continuous variable, indicating the price each passenger paid to get onboard the ship. This column is possibly strongly related to the `PClass` column, discussed previously and it is very likely a strong indicator for the survival rate. Once again, we can visualise the continuous variables by using bins and grouping the fares at certain ranges. As an example, the illustration below shows that we have ended up with 4 bins, where it can clearly be seen that the higher the price, the higher the survival rate. Note that we have used a quantile-based discretization function, which illustrates that the majority of tickets are priced between 0.001 and 31, while only a few passengers paid more (of course, this was also expected). The test data has a few missing values under this column. Due to the multiple outliers, it is likely a good idea to use a median-based inputing to fill in the missing data.  
   ![alt text](img/fare_rel.png)
10. `Cabin` indicates the cabin number the passenger had on board. Upon inspection, it can be seen that some passengers had multiple cabins, possibly due to the fact they were a part of a larger family. Since most passengers likely had different cabins, this feature is merely another (almost) unique identifier for each passenger. With this said, it is highly unlikely that it will add a lot of significance to the model.
11. `Embarked` is a categorical variable, incidating the place of onboarding for each passenger. _C_ stands for _Cherbourg_, _Q_ stands for _Queenstown_ and _S_ stands for _Southampton_. As we can see on the illustration below, many more passengers that embarked at _Cherbourg_ eventually survived.
    ![alt text](img/embarked_rel.png)
