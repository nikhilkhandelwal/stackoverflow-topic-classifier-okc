1) What metric did you use to judge your approach’s performance, and how did it perform?
Why did you choose that metric?

      #### I used precision , recall , and F1 score to judge the models that I built.

2) The dataset we’ve given you is artificially balanced such that there’s an even split of
closed posts to accepted posts. Should this influence the metrics you measure?

      #### Yes, this means that the real test dataset will have a different distribution. I have used the ensemble learning model Random Forrest and ADA Boost which  are robust against over fitting and should perform better on the imbalanced dataset.
3) How generalizable is your method? If you were given a different (disjoint) random
sample of posts with the same labeling scheme, would you expect it to perform well?
Why or why not? Do you have evidence for your reasoning?

      #### The model used is Random Forrest ,which are is a meta ensemble of of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. So the model should generalize well.

4) How well would this method work on an entirely new close reason, e.g. duplicate or
spam posts?
      ####  Depends on the reason, if the  text embeddings are able to differentiate between classes, then it might perform ok , but features like "code block", "number of urls" might not be applicable

5) Are there edge cases that your method tends to do worse with? Better? E.g., How well
does it handle really long posts or titles?

     #### Very short questions, are tough to classify
6) If you got to work on this again, what would you do differently (if anything)?
     #### Since cnn had good results, i would like to add the numeric features and train a more complex model.
     #### I would also want to do more feature engineering 
7) If you found any issues with the dataset, what are they?
    #### I didnt find any.