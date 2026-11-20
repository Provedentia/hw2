# CS396 Fairness in ML Homework 2

This assignment is worth 10 points total (not including bonus points).
Deadlines and the late day policy are listed on Canvas. Please put your Net ID
in the `netid` file.

## Academic integrity

This is an individual assignment. You may not share or receive code or written
materials with other students in any circumstances. You may use Copilot or LLMs
for coding, but you may not use LLMs to produce or revise your write-up. We
will not deduct points for minor grammatical errors.

## Write-up requirements

You will submit to Canvas a write-up that should be no more than two pages
long, not including references. This write-up should describe the models you
trained and how you evaluated them. A list of specific questions your write-up
should answer is in `questions.md`.

## Leaderboard and prediction files

While we will not directly grade your code or any models you train, you will
need to train models that make predictions on the provided test set features.
Use the provided `hw2.csv.gz` and `hw2_test.csv.gz` files (see below). Your written
report should discuss the performance of your models on both the train and test sets.

To submit a model (or baseline) to be evaluated on the test set, you must push
a csv file to the `predictions/` folder. That file  should have two columns
with headers of "PREDICTION" and "PROBABILITY". See `predictions/example.csv`
for an example of the expected format.

We will run the model evaluation script every hour. For each csv file uploaded,
if it has been changed since the most recent commit, we will reevaluate those
predictions and add a new entry to the public (pseudonymous) leaderboard.

Every time the autograder evaluates models, it will create a new csv file of
the results of all students' submissions across a variety of metrics. These
results will be public to all students in the class, but your identity is
hidden behind a pseudonym.  For example, if the autograder ran on November 6 at
10am on a predictions file `baseline7.csv` that you added in a commit with hash
`abcd1234`, and your `pseudonym` file contains the text `hello`, then your
result on the leaderboard will be displayed as:
`2026-11-06-hello-baseline7-abcd1234`.

We will award at least one bonus point to the student with the highest score in
each metric tracked by the autograder, except in cases where the provided
baselines are perfect on those metrics.


## Allowed packages

You can use whatever packages you want locally, but your submitted model file
must run on the autograder, which will have a limited number of packages
installed. You may request that we install a new package on the autograder,
but we cannot guarantee our ability to do so. At the time of releasing the
assignment, the autograder has the following packages installed:

- `fairlearn`
- `mapie`
- `numpy`
- `pandas`
- `scikit-learn`
- `scipy`
- `torch`

## Autograder metrics

The metrics reported by the autograder leaderboard will be given unhelpful
placeholder names. In your write-up (see `questions.md`), you will try to
decipher which metric is which. To do this, you should carefully evaluate your
model on the provided data (use cross validation or held-out test sets!) and
then use your leaderboard results to guess. You don't have to guess perfectly
to earn full points, but you need to describe your approach. You are allowed to
submit predictions with intentionally bad performance in order to gather information
about the autograder's metrics.

We will use at least some of the following metrics, from these packages:

- `sklearn.metrics.accuracy_score`
- `sklearn.metrics.f1_score`
- `sklearn.metrics.precision_score`
- `sklearn.metrics.recall_score`
- `fairlearn.metrics.demographic_parity_difference`
- `fairlearn.metrics.equalized_odds_difference`
- `mapie.metrics.calibration.expected_calibration_error`

We may also use `between_group_calibration` from the provided `calibration.py`
file.

## Dataset

We provide a `hw2.csv.gz` and `hw2_test.csv.gz` dataset on [`Canvas > Files >
HW2`](https://canvas.northwestern.edu/courses/258037/files/folder/HW2). You
can load `gzip` compressed files directly with `pandas`, but you can `gzip -d`
them to decompress them if you want them in plaintext.

This is a binary classification task. You should try to predict `LABEL` from
the other features. The dataset has two protected attributes ('PAs') for which
we would like to be fair. The dataset has four other features. The format of
`hw2.csv.gz` and `hw2_test.csv.gz` are the same, except that `LABEL` is removed
from the test set.

After the assignment is due, I will share a link to the original dataset and
its documentation. Please do not try to figure out the origin of the dataset
before then.

## Questions?

Please ask on Piazza. We will pin a post with clarifications or corrections
about the assignment.
