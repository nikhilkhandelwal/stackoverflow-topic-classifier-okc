1) What metric did you use to judge your approach’s performance, and how did it perform?
Why did you choose that metric?

      #### I used precision , recall , and F1 score to judge the models that I built.

2) The dataset we’ve given you is artificially balanced such that there’s an even split of
closed posts to accepted posts. Should this influence the metrics you measure?

      #### Yes, we should use a precision recall curve to measure the performance on imbalanced data. Precision is influenced by class imbalance, we should use PR curve to compare different models and approaches on this dataset.
      
3) How generalizable is your method? If you were given a different (disjoint) random
sample of posts with the same labeling scheme, would you expect it to perform well?
Why or why not? Do you have evidence for your reasoning?

     #### The model I used is Random Forrest ,which is a meta ensemble of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. So the model should generalize well.

4) How well would this method work on an entirely new close reason, e.g. duplicate or
spam posts?
      ####  Depends on the reason, if the  text embeddings are able to differentiate between classes, then it might perform ok , but features like "code block", "number of urls" might not be applicable
      #### Will not work on duplicate tagging problems
5) Are there edge cases that your method tends to do worse with? Better? E.g., How well
does it handle really long posts or titles?

     #### Very short questions are tough to classify
6) If you got to work on this again, what would you do differently (if anything)?
     #### Since cnn had good results, I would like to add the numeric features and train a more complex model.
     #### I would also want to do more feature engineering 
     #### Add test cases
7) If you found any issues with the dataset, what are they?
    #### Since off topic itself is subjective, so there were some similar questions with opposite labels. So I think label noise might be an issue
