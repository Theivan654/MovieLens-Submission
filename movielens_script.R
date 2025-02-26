# Load necessary libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(reshape2)

# Set timeout options for downloading large files
options(timeout = 120)

# Define the path to the MovieLens dataset
dataset_path <- "C:/RSTUDIO/MOVIE LENS"

# Load the data (adjust the path if needed)
ratings_file <- file.path(dataset_path, "ml-10M100K/ratings.dat")
movies_file <- file.path(dataset_path, "ml-10M100K/movies.dat")

# Read the ratings data and process it
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Read the movies data and process it
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

# Join ratings and movie data
movielens <- left_join(ratings, movies, by = "movieId")

# Create training and hold-out test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# Build the basic recommendation model (Mean rating)
avg_rating <- mean(edx$rating)
predictions <- edx %>%
  mutate(predicted_rating = avg_rating)

# Calculate RMSE for the basic model
rmse_basic <- sqrt(mean((predictions$rating - predictions$predicted_rating)^2))
cat("RMSE with basic model: ", rmse_basic, "\n")

# Build model with movie bias
movie_bias <- edx %>%
  group_by(movieId) %>%
  summarize(b_movie = mean(rating - avg_rating))

predictions_movie_bias <- edx %>%
  left_join(movie_bias, by = "movieId") %>%
  mutate(predicted_rating = avg_rating + b_movie)

# Calculate RMSE with movie bias
rmse_movie_bias <- sqrt(mean((predictions_movie_bias$rating - predictions_movie_bias$predicted_rating)^2))
cat("RMSE with movie bias: ", rmse_movie_bias, "\n")

# Build model with user and movie bias
user_bias <- edx %>%
  left_join(movie_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_user = mean(rating - avg_rating - b_movie))

predictions_user_movie_bias <- edx %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(predicted_rating = avg_rating + b_movie + b_user)

# Calculate RMSE with user and movie bias model
rmse_user_movie_bias <- sqrt(mean((predictions_user_movie_bias$rating - predictions_user_movie_bias$predicted_rating)^2))
cat("RMSE with user and movie bias: ", rmse_user_movie_bias, "\n")

# Output the final RMSE for the user and movie bias model
cat("Final RMSE with user and movie bias: ", rmse_user_movie_bias, "\n")

# Final Evaluation: Use final_holdout_test set to evaluate final RMSE
final_predictions <- final_holdout_test %>%
  left_join(movie_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(predicted_rating = avg_rating + b_movie + b_user)

final_rmse <- sqrt(mean((final_predictions$rating - final_predictions$predicted_rating)^2))
cat("Final RMSE on holdout test set: ", final_rmse, "\n")
