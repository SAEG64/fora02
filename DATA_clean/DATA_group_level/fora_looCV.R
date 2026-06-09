# Load necessary libraries
# library(lme4)
library(brms)
library(dplyr)
library('readr')
library(ROCR)
library(ggplot2)
# library(lme4)
# Set working directory
set.seed(123)
filepath <- paste(dirname(rstudioapi::getSourceEditorContext()$path), "/", sep = "")
setwd(filepath)
# Data import and filtering
dat <- read_csv("test_data.group_level_univar.csv")
dat <- subset(dat, !is.na(key_resp.keys)) # exclude none responses
dat <- filter(dat, in_LP > 0) # exclude starvation states
dat <- filter(dat, condition_rORp == "high threat condition") # filter approach/avoidance condition

# Function to perform leave-one-out cross-validation
loo_cv <- function(data, file_name) {
  
  # Initialize requirements
  groups = unique(dat$Subject_ID)
  n <- length(groups)
  preds <- c()
  coefs <- c()
  
  # Loop through each observation
  for (i in 1:n) {
    # Create training data by excluding the data from one subject
    train_data <- data[data$group != groups[i],]
    test_data <- data[data$group == groups[i], ]
    # Remove the 'subject' column from the test data
    test_data <- test_data[, !names(test_data) %in% "group", drop = FALSE]
    
  ##############################################################################
    # Fit the logistic mixed effects model (logit link)
  ##############################################################################
    # model <- glmmLasso(y ~ (
    #   x1 +
    #     x2 +
    #     (1 | group)
    #   ),
    #   data = train_data,
    #   family="binomial"(link = "logit"),
    #   control = glmerControl(optimizer="bobyqa"))
    
    # Fit a Bayesian logistic regression model with random intercepts
    model <- brm(
      formula = y ~ (
          x1 +
            x2 +
            (1 | group)
          ),
      family = bernoulli(),
      prior = c(
        prior(normal(0, 5), class = "b"), # Normal prior for fixed effects
        prior(normal(0, 5), class = "sd")  # Normal prior for random effects
      ),
      data = data,
      cores = 4
    )
    
    # Make predictions on the test set, excluding random effects (use only fixed effects)
    # preds <- c(preds, 
    #                  predict(model, newdata = test_data, 
    #                          type = "response", re.form = NA))
    # Posterior predictions of the response variable
    post_preds <- predict(model, summary = TRUE)
    # Accessing the mean (point estimate) and intervals
    preds <- preds[, "Estimate"] # means
    # preds_lower <- preds[, "Q2.5"]
    # preds_upper <- preds[, "Q97.5"]    
    
    # Extract the fixed effects coefficients and store them
    fixed_effects <- fixef(model)  # Fixed-effect coefficients
    coefs <- c(coefs, list(as.numeric(fixed_effects)))
  }

  # Compute accuracy and area under the curve (ac)
  randomized_vector <- runif(length(preds), min = 0, max = 1)
  pred_class <- ifelse(runif(length(randomized_vector)) <= preds, 1, 0)
  true_class <- data$y
  # # Accuracy
  # accuracy <- mean(pred_class == true_class)
  # AUC-ROC
  pred <- prediction(pred_class, true_class)
  ac <- performance(pred, "auc")@y.values[[1]]
  
  ##############################################################################
  # Visualize predictions
  ##############################################################################
  # Prep data fro visualization
  plot_data <- data.frame(
    GroundTruth = as.numeric(true_class),
    Predicted = as.numeric(preds),
    x1 = data$x1,
    x2 = data$x2,
    group == data$group
  )
  if (is.factor(plot_data$x2) == FALSE) {
    plot_data$bin <- cut(plot_data$x2, breaks = 9)  # Adjust number of bins as needed
  }else {
    plot_data$bin <- plot_data$x2  # Adjust number of bins as needed
  }
  # Aggregate the count of occurrences per bin and calculate average predicted probabilities for each bin
  bin_counts_sbj <- plot_data %>%
    group_by(bin, x1, group) %>%
    summarize(
      count = n(),  # Count the occurrences in each bin
      avg_pred_prob = mean(Predicted),
      avg_prob = mean(GroundTruth),
      #.groups = 'drop'
    )
  bin_counts <- bin_counts_sbj %>%
    group_by(bin, x1) %>%
    summarize(
      count = mean(count),  # Count the occurrences in each bin
      avg_pred_prob = mean(Predicted),
      avg_prob = mean(GroundTruth),
      .groups = 'drop'
    )
  data_merged <- merge(plot_data, bin_counts, by = "bin", all.x = TRUE)
  # Create the plot
  p <- ggplot(bin_counts, aes(x = bin, y = avg_pred_prob, 
                              group = x1, color = x1, , linetype = x1)) +
    # Plot the predicted logistic curve
    geom_line(size = 1,  linetype = "dashed") +
    geom_point(size = 2, shape = 5) +
    # Plot the actual observed responses as bubbles, scaled by the count in each bin
    geom_point(data = bin_counts, aes(x = bin, y = avg_prob, size = as.numeric(count)),
                alpha = 0.7) +
    # Customize the plot
    ylim(0, 1) +  # Set y-axis limits from 0 to 1
    labs(x = "binned model", y = "Foraging likelihood", title = paste("ternary state +", toString(mdl[j]))) +
    scale_size_continuous(name = "Aggregated responses", range = c(3, 10)) +  # Adjust size range
    theme_minimal(base_size = 15) + # Adjust 'base_size' for proportional scaling
    theme(
      axis.text = element_text(size = 14),
      axis.title = element_text(size = 16),
      legend.text = element_text(size = 14),
      legend.title = element_text(size = 16)
    )
    theme_minimal()
    # theme(legend.position = "none")
  # Save plot
  if (!is.null(file_name)) {
    ggsave(file_name, plot = p)
  }

  return(list(accuracy_metric = ac, coefficients = coefs, predictions = preds, posterior_predictions = post_preds))
}

# Models to compare
mdl <- c(
  # 'win stay lose shift',
  # '** wait when safe',
  # '** binary energy state',
  # 'weather type',
  # '* $\\mathit{r}$ predator',
  '* $\\mathit{p}$ gain',
  'expected gain naive',
  '** $\\mathit{p}$ success',
  # 'marginalC value',
  # '$\\mathit{p}$ success + BES',
  # '$\\mathit{p}$ success + WWS',
  # 'multi-heuristic policy')#,
  # '$\\mathit{OP}$ values + cap')#,
  'optimal policy values')

# Raw model components
resp <- dat$fora_response
mod1 <- as.numeric(unlist(dat['ternary state']))
# mod1 <- ifelse(mod1 > 0.5, 3, mod1)
# mod1 <- ifelse(mod1 < 0.5, 1, mod1)
# mod1 <- ifelse(mod1 == 0.5, 2, mod1)

# For saving results of LOO-CV
all_results = list()
acs <- c()
# Loop over models
for (j in 1:length(mdl)) {
  
  cat("Current DV: ", mdl[j], "\n")
  
  # Decision variables
  x1 = mod1             # Fixed predictor
  # Ensure dummy coding for factors
  if (length(unique(x1)) < 4) {
    x1 = factor(x1)
  }
  x2 = c(dat[mdl[j]])[[1]]   # Variable predictor
  # Ensure dummy coding for factors
  if (length(unique(x2)) < 4) {
    x2 = factor(x2)
  }
  
  # Define data
  data <- data.frame(
    y = resp,   # Binary outcome variable
    x1 = x1,             # Fixed predictor
    x2 = x2,   # Variable predictor
    group = factor(dat$Subject_ID)  # Random effect grouping variable
  )
  colnames(data) <- c("y", "x1", "x2", "group")
  # Removes rows with any NA or NaN values
  data <- data[complete.cases(data), ]
  
  # Name for plot saving
  file_name = paste("multivar_fit_", toString(mdl[j]), ".png", sep="")
  
  # Run the LOO-CV
  res <- loo_cv(data, file_name)
  all_results[[j]] = res
  acs <- c(acs, res[1])
  
  # Output the accuracy
  cat("Leave-One-Out Cross-Validation accuracy metric: ", toString(res[1]), "\n")
  print("=================================")
}

cat('\nbest DV:   ', mdl[order(unlist(acs), decreasing=TRUE)[1]],
    " accuracy metric: ", toString(acs[order(unlist(acs), decreasing=TRUE)[1]]),'\n')
cat('\nsecond DV: ', toString(mdl[order(unlist(acs), decreasing=TRUE)[2]]),
    " accuracy metric: ", toString(acs[order(unlist(acs), decreasing=TRUE)[2]]),'\n')
cat('\nthird DV:  ', mdl[order(unlist(acs), decreasing=TRUE)[3]],
    " accuracy metric: ", toString(acs[order(unlist(acs), decreasing=TRUE)[3]]),'\n')
print("=================================")
cat('Are second and third best DV ac values identical?\n--->', identical(
  acs[order(unlist(acs), decreasing=TRUE)[2]], 
  acs[order(unlist(acs), decreasing=TRUE)[3]]))
